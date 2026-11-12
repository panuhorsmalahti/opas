# Running opastin on Kubernetes (plain manifests)

`deploy/kubernetes/` is a Kustomize base containing:

- a Deployment;
- a ClusterIP Service;
- a ConfigMap with `opastin.toml`.

It runs the random router. For the probe router, GPU nodes and model storage, the
[Helm chart](helm.md) has the options built in.

## Install

```sh
kubectl create namespace opastin
kubectl -n opastin create secret generic opastin-api-keys \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
kubectl -n opastin apply -k deploy/kubernetes
kubectl -n opastin rollout status deploy/opastin
```

The Secret is optional (`optional: true`). Without it, clients' own credentials are
forwarded upstream.

## Customize

Make an overlay rather than editing the base:

```yaml
# my-overlay/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: opastin
resources:
  - ../deploy/kubernetes
images:
  - name: opastin
    newName: ghcr.io/OWNER/opastin
    newTag: "0.1.0"
configMapGenerator:
  - name: opastin-config
    behavior: replace
    files:
      - opastin.toml
```

```sh
kubectl apply -k my-overlay
```

After changing the ConfigMap, restart the pods: `kubectl -n opastin rollout restart
deploy/opastin`. The config is read at startup.

## Use it from inside the cluster

```text
http://opastin.opastin.svc.cluster.local/v1          # OpenAI SDKs, Open Responses
http://opastin.opastin.svc.cluster.local             # Anthropic SDKs
http://opastin.opastin.svc.cluster.local/v1beta      # Gemini Interactions
```

To try it from your machine: `kubectl -n opastin port-forward svc/opastin 8080:80`.

## Scaling and state

Cache-aware routing and provider-state affinity (`previous_response_id` /
`previous_interaction_id`) are tracked in memory in each pod. If a conversation's turns
land on different pods, routing still works, but:

- **Cache awareness is lost:** a pod that did not serve the previous turn does not know
  which model's prompt cache is warm.
- **Stored-state follow-ups may break:** a follow-up can go to a model that does not hold
  the referenced state, and that provider will reject it.

So:

- The Service uses `sessionAffinity: ClientIP`. This works when each client has a stable
  source IP, but not behind a NAT shared by many clients.
- With an ingress controller, prefer cookie- or header-based stickiness (for example
  hashing on an API key or user header).
- One replica is enough for the proxy and random router at high request rates. The
  work is forwarding, not computing.

## Probe router on Kubernetes

The probe router needs the encoder model file (9.5 GB) and roughly 11 GiB of memory per
pod. On CPU nodes it adds seconds per routed request; GPU nodes with the CUDA image are
the practical choice. See [hardware](hardware.md).

The Helm chart handles the extra pieces:

- a PVC;
- an init container that downloads the model;
- GPU resource limits and a `runtimeClassName`.

The fastest option without GPU nodes is to run the probe router natively on Apple Silicon
hosts outside the cluster, and expose them as a Service with manual Endpoints:

```yaml
apiVersion: v1
kind: Service
metadata: {name: opastin-probe, namespace: opastin}
spec:
  ports: [{name: http, port: 80, targetPort: 8080}]
---
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: opastin-probe-1
  namespace: opastin
  labels: {kubernetes.io/service-name: opastin-probe}
addressType: IPv4
ports: [{name: http, port: 8080}]
endpoints:
  - addresses: ["10.0.0.42"]     # the Mac running `opastin --config opastin.toml`
```
