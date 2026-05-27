# rules_lora — internal notes

**Status:** private / fastverk premium tier. Do not push to GitHub,
do not register in any public bazel-registry, do not reference
publicly.

## What this is

Bazel-native LoRA fine-tuning. The user-facing surface is four
macros:

| macro | what it declares |
|---|---|
| `lora_dataset` | A typed SFT dataset (JSONL) with build-time validation: schema, min-example count, base-model compatibility |
| `lora_recipe`  | A declarative training recipe — base model, LoRA hyperparameters, training-loop knobs. Renders to torchtune / axolotl / peft YAML at execute time |
| `lora_train`   | The training action. `bazel run :<name>.train` orchestrates a backend (local CPU smoke, RunPod H100, future: Modal / SkyPilot) and stages the resulting `.safetensors` adapter |
| `expert_manifest` | Bundles N trained adapters into an `ExpertManifest.binpb` — the routing contract from `[[rules_agentic_ide]]/proto/agentic_ide/v1/experts.proto` |

```starlark
load("@rules_lora//lora:defs.bzl",
     "lora_dataset", "lora_recipe", "lora_train", "expert_manifest")

lora_dataset(
    name = "cluster_9_sft",
    src  = "training/shards/cluster_009/sft.jsonl",
    schema = "messages_v1",
    min_examples = 50,
)

lora_recipe(
    name = "rank16_a32_3epoch",
    framework = "torchtune",
    rank = 16,
    alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    learning_rate = 2e-4,
    micro_batch_size = 4,
    grad_accum_steps = 8,
    epochs = 3,
)

lora_train(
    name = "cluster_9",
    base = "@hf//google/gemma-3-2b-it",  # or a lora_base_model() target
    recipe = ":rank16_a32_3epoch",
    dataset = ":cluster_9_sft",
    backend = "runpod",
)
```

## Why this exists

Three first-party projects need LoRA training with the same
input-output shape:

* `[[rules_agentic_ide]]` — per-cluster LoRAs over the user's
  Claude-Code chat corpus, one expert per k-means cluster of the
  BGE-embedding manifold. See `proto/agentic_ide/v1/experts.proto`
  (the `ExpertManifest` contract this repo emits).
* `[[agora]]` — per-tool bid models. Each tool in the capability
  auction publishes a small fine-tuned model that scores its own
  ability to satisfy a query. Same training shape; different
  routing.
* A third (operator's NDA'd) project also pending.

Without this repo, each of the three hand-rolls runpod
orchestration, torchtune wiring, HF model caching, adapter
staging. Lift once.

## Layout

```
lora/
  defs.bzl                 public macros (lora_dataset / _recipe / _train / expert_manifest)
  private/
    providers.bzl          LoraDatasetInfo, LoraRecipeInfo, LoraAdapterInfo
    rules.bzl              rule(impl=...) for each macro
    backends/
      runpod.bzl           shells out to runpod_orchestrator
      local.bzl            local CPU smoke (tiny model only)
runtime/
  torchtune_runner/        Python entrypoint that runs inside the training container
  runpod_orchestrator/     Rust binary — uploads recipe+dataset, polls job, downloads .safetensors
  hf_resolver/             Rust — hf-hub revision -> sha256 digest, caches to a content-addressed local dir
proto/
  lora.proto               TrainingJobSpec, AdapterArtifact, ExpertManifest envelope
examples/
  smoke/                   tiny model + tiny dataset + local backend smoke test (CPU, fast)
```

## Non-goals (v0)

* **Full pretraining.** This repo is LoRA-only. Full FT is a
  different recipe family with different infra costs.
* **Hermetic training inside a Bazel action.** Training calls out
  to an H100 — Bazel can't sandbox a remote GPU. The recipe and
  dataset are Bazel artifacts; the trained adapter is staged back
  in via the train rule's output. Same pattern rules_tectonic
  uses for LaTeX → PDF.
* **Per-step reproducibility.** Adapters are pinned by recipe sha
  + dataset sha + base-model digest, but exact-bit reproducibility
  depends on backend determinism (cuDNN, etc.) which we don't
  promise.

## Status

v0.0.1 — scaffold. Macro signatures decided, providers stubbed,
runner binaries are TODOs. First milestone: `lora_train` against
the local CPU backend with a 0.5B-parameter base model and a
20-example dataset; sub-second-per-step, smoke-only.
