# Changelog

All notable changes to rules_lora. The format is loosely
[Keep a Changelog](https://keepachangelog.com/) — version headers
mirror the published bazel-registry entries (when we publish; for
now this repo is premium / private).

## 0.0.1 — scaffold + public API frozen

Public surface (`@rules_lora//lora:defs.bzl`):

* `lora_dataset` — typed SFT-JSONL dataset, validated + sha-pinned
  at build time. Schemas: `messages_v1` (OpenAI chat format),
  `instruction_v1`.
* `lora_recipe` — declarative training recipe. Frameworks:
  `torchtune` (rendered), `axolotl` (rendered), `peft` + `trl`
  (TODO templates).
* `lora_base_model` — HF hub model pinned by repo + revision.
* `lora_train` — composes the inputs into a
  `lora.v1.TrainingJobSpec` (JSON, build-time) that a backend
  executes at `bazel run` time.
* `expert_manifest` — bundles N adapters into the
  `agentic_ide.v1.ExpertManifest.binpb` shape (placeholder
  filegroup in v0.0.1; real rule in v0.0.2).

Runtime stubs:

* `runtime/torchtune_runner/{validate_jsonl, render_recipe}.py` —
  build-time tools, std-lib only.
* `runtime/runpod_orchestrator/` (Rust) — `write-jobspec`
  subcommand functional; `run` subcommand pending v0.1.

Smoke at `examples/smoke/` exercises all four macros end-to-end
and produces a self-contained jobspec — `bazel build
//examples/smoke:smoke_jobspec` is the regression test.
