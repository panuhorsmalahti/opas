# Running opas on Kubernetes (plain manifests)

`deploy/kubernetes/` is a Kustomize base containing:

- a Deployment;
- a ClusterIP Service;
- a ConfigMap with `opas.toml`.

It runs the random router. For the probe router, GPU nodes and model storage, the
[Helm chart](helm.md) has the options built in.

## Install

```sh
kubectl create namespace opas
kubectl -n opas create secret generic opas-api-keys \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
kubectl -n opas apply -k deploy/kubernetes
kubectl -n opas rollout status deploy/opas
```

The Secret is optional (`optional: true`). Without it, clients' own credentials are
forwarded upstream.

## Customize

Make an overlay rather than editing the base:

```yaml
# my-overlay/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: opas
resources:
  - ../deploy/kubernetes
images:
  - name: opas
    newName: ghcr.io/OWNER/opas
    newTag: "0.1.0"
configMapGenerator:
  - name: opas-config
    behavior: replace
    files:
      - opas.toml
```

```sh
kubectl apply -k my-overlay
```

After changing the ConfigMap, restart the pods: `kubectl -n opas rollout restart
deploy/opas`. The config is read at startup.

## Use it from inside the cluster

```text
http://opas.opas.svc.cluster.local/v1          # OpenAI SDKs, Open Responses
http://opas.opas.svc.cluster.local             # Anthropic SDKs
http://opas.opas.svc.cluster.local/v1beta      # Gemini Interactions
```

To try it from your machine: `kubectl -n opas port-forward svc/opas 8080:80`.

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
metadata: {name: opas-probe, namespace: opas}
spec:
  ports: [{name: http, port: 80, targetPort: 8080}]
---
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: opas-probe-1
  namespace: opas
  labels: {kubernetes.io/service-name: opas-probe}
addressType: IPv4
ports: [{name: http, port: 8080}]
endpoints:
  - addresses: ["10.0.0.42"]     # the Mac running `opas --config opas.toml`
```

## Model prices

opas fetches model prices from models.dev (`https://models.dev/api.json`) at startup and
once a day. If pods have no outbound internet access:

- set prices per model in the config (`pricing = {…}`), or
- point `[pricing] models_dev_url` at an internal mirror, or
- disable the lookup with `models_dev = false`; tier default prices are then used.

Check what is in effect with `GET /v1/pricing`.
