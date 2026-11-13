# Running opas on Kubernetes (plain manifests)

`deploy/kubernetes/` is a Kustomize base containing:

- a Deployment;
- a ClusterIP Service;
- a ConfigMap with `opas.toml`;
- a 20 Gi PersistentVolumeClaim for the encoder model.

It runs the probe router. The predictor ships in the image; an init container downloads
the encoder model (Qwen3.5-9B Q8_0, 9.5 GB) into the PVC on the first start. The pod
requests 4 CPUs and 12 Gi of memory and runs the encoder on CPU. For GPU nodes and more
options, use the [Helm chart](helm.md).

For a plain proxy (clients name a model, no routing), replace the ConfigMap with a
config without `[router]` and drop the PVC, the init container and the large resource
requests in an overlay.

## Install

```sh
kubectl create namespace opas
kubectl -n opas create secret generic opas-api-keys \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
kubectl -n opas apply -k deploy/kubernetes
kubectl -n opas rollout status deploy/opas --timeout=30m
```

The first start downloads the 9.5 GB encoder model (follow it with
`kubectl -n opas logs deploy/opas -c fetch-models -f`). Later restarts reuse the PVC.

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
    newName: ghcr.io/panuhorsmalahti/opas
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
- Each replica loads its own copy of the encoder (about 11 GiB). With the default
  `ReadWriteOnce` PVC, replicas must share a node; use `ReadWriteMany` storage or a
  PVC per replica (a StatefulSet) to spread them.

## Probe router on Kubernetes

On CPU nodes the encoder adds seconds per routed request; GPU nodes with the CUDA image
are the practical choice. See [hardware](hardware.md). The Helm chart adds GPU resource
limits and a `runtimeClassName`.

If the cluster cannot reach Hugging Face, pre-populate the PVC with
`gguf/Qwen3.5-9B-Q8_0.gguf`; the init container skips files that are already there.

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
