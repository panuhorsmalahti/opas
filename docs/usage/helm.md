# Installing opas with Helm

The chart is in `deploy/helm/opas`.

## Install

```sh
kubectl create namespace opas
kubectl -n opas create secret generic opas-api-keys --from-literal=ANTHROPIC_API_KEY=sk-ant-...

helm install opas deploy/helm/opas -n opas \
  --set image.repository=ghcr.io/OWNER/opas \
  --set apiKeys.existingSecret=opas-api-keys \
  --set-file config=opas.toml
```

`--set-file config=...` uses your own `opas.toml`; without it the chart's example
config is used (random choice between two Claude models). After the install, `helm status`
prints how to try it with a port-forward.

## Values

| Value | Default | Purpose |
|---|---|---|
| `image.repository`, `image.tag` | `ghcr.io/OWNER/opas`, chart appVersion | Image. Use the `Dockerfile.cuda` build for GPU nodes |
| `config` | example TOML | The full `opas.toml`. Changing it rolls the pods (checksum annotation) |
| `apiKeys.existingSecret` | `""` | Secret whose keys become environment variables (the names in `api_key_env`) |
| `apiKeys.values` | `{}` | Or let the chart create that Secret from these values |
| `replicaCount` | `1` | See "Scaling" below |
| `service.type`, `service.port` | `ClusterIP`, `80` | Service |
| `service.sessionAffinity` | `ClientIP` | Keeps a client's conversations on one pod (routing state is per pod) |
| `ingress.*` | disabled | Standard Ingress |
| `models.persistence.enabled` | `false` | PVC mounted at `models.mountPath` (`/models`) for the probe router |
| `models.persistence.existingClaim`, `size`, `storageClass`, `accessModes` | `""`, `20Gi`, `""`, `[ReadWriteOnce]` | PVC settings |
| `models.download` | `[]` | `{url, path}` files an init container fetches into the volume when missing |
| `gpu.enabled`, `gpu.resource`, `gpu.count` | `false`, `nvidia.com/gpu`, `1` | Requests a GPU for the encoder |
| `runtimeClassName` | `""` | e.g. `nvidia` on clusters that use a GPU runtime class |
| `resources` | 100m CPU / 64Mi requested, 512Mi limit | Raise them for the probe router (see below) |
| `extraEnv`, `podAnnotations`, `podLabels`, `nodeSelector`, `tolerations`, `affinity` | empty | Usual pod settings |
| `podSecurityContext`, `securityContext` | non-root, read-only root filesystem, no capabilities | Hardened defaults |

## Example: probe router on NVIDIA GPU nodes

```yaml
# values-probe-gpu.yaml
image:
  repository: ghcr.io/OWNER/opas
  tag: cuda                       # built from Dockerfile.cuda
apiKeys:
  existingSecret: opas-api-keys
gpu:
  enabled: true
nodeSelector:
  nvidia.com/gpu.present: "true"
resources:
  requests: {cpu: "2", memory: 12Gi}
  limits: {memory: 14Gi}
models:
  persistence:
    enabled: true
    size: 20Gi
  download:
    - url: https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf
      path: gguf/Qwen3.5-9B-Q8_0.gguf
    - url: https://YOUR-STORAGE/llmrouterbench-qwen35-9b.json   # the probe you trained
      path: probes/llmrouterbench-qwen35-9b.json
config: |
  router_model = "auto"

  [router]
  kind = "probe"
  encoder_model = "/models/gguf/Qwen3.5-9B-Q8_0.gguf"
  probe = "/models/probes/llmrouterbench-qwen35-9b.json"
  lambda = 10.0

  [[models]]
  id = "claude-haiku-5-5"
  tier = "small"
  anthropic = { base_url = "https://api.anthropic.com/v1", api_key_env = "ANTHROPIC_API_KEY" }

  [[models]]
  id = "claude-opus-5-5"
  tier = "large"
  anthropic = { base_url = "https://api.anthropic.com/v1", api_key_env = "ANTHROPIC_API_KEY" }
```

```sh
helm upgrade --install opas deploy/helm/opas -n opas -f values-probe-gpu.yaml
```

**First start:**

- The init container downloads 9.5 GB. Later restarts reuse the PVC.
- Loading the model takes a few seconds; the startup probe allows up to 10 minutes.
- For the downloads to work, the cluster must reach Hugging Face. Otherwise
  pre-populate the PVC and leave `models.download` empty.

**On CPU-only nodes:**

- Drop `gpu` and `image.tag: cuda`, and request plenty of whole cores (`cpu: "8"` or
  more).
- Expect seconds of added latency per routed request; see [hardware](hardware.md).

## Scaling

Routing state (cache-aware conversation tracking, and affinity for
`previous_response_id` / `previous_interaction_id`) lives in each pod's memory:

- Keep `replicaCount: 1`, or keep `service.sessionAffinity: ClientIP`, or use sticky
  sessions at the ingress.
- Each probe-router replica loads its own copy of the encoder, about 11 GiB of memory and
  one GPU each.

## Upgrade and uninstall

```sh
helm upgrade opas deploy/helm/opas -n opas -f my-values.yaml
helm uninstall opas -n opas
```

The models PVC is annotated `helm.sh/resource-policy: keep`, so uninstalling keeps the
downloaded models. Delete it yourself when you no longer need it:
`kubectl -n opas delete pvc opas-models`.
