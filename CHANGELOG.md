# Changelog

All notable changes to rules_lora. The format is loosely
[Keep a Changelog](https://keepachangelog.com/) — version headers
mirror the published bazel-registry entries (when we publish; for
now this repo is premium / private).

## 0.0.3 — `lora_corpus` rule with corpus-DAG deps

New public macro `lora_corpus`: declare an SFT dataset *produced by
running a user-supplied transform binary over a `source` filegroup,
chained from upstream corpora via `deps`*. Three consumers
(rules_agentic_ide chat traces, agora capability-auction corpus,
the NDA'd third) share the input/transform/validate/output skeleton;
the rule factors it out.

Public surface added:

* `lora_corpus(name, source, transform, deps, schema, min_examples)`
  — runs the transform once with repeated `--input` / `--corpus-dep`
  / single `--output` flags, then runs the existing
  `validate_jsonl` validator on the transform output. Returns
  `LoraDatasetInfo`, so it plugs in anywhere `lora_dataset` is
  accepted (in particular as the `dataset` attr of `lora_train`).

Corpus deps form a DAG. The rule itself flattens to *direct* deps
when invoking the transform (the upstream corpora's transforms
have already produced their validated JSONL artifacts as build
outputs); Bazel's build graph enforces no cycles and propagates
transitive rebuilds.

Smoke at `examples/corpus_smoke/`:

    bazel build //examples/corpus_smoke:derived_corpus
    # base_corpus: 3 examples
    # derived_corpus: 6 examples (3 source + 3 from dep)

Also includes the v0.0.2 features (deferred from registry release):
`lora_train` macro composes with `@rules_runpod` when
`backend = "runpod"`, auto-emitting `<name>_runpod_job.run`.

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
