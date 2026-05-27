"""rules_lora private rule implementations.

The macros in //lora:defs.bzl are thin wrappers that pick a backend
and instantiate these rules. Keeping the rule defs private lets us
evolve the public surface without touching every consumer.
"""

load(":providers.bzl",
     "LoraAdapterInfo",
     "LoraBaseModelInfo",
     "LoraDatasetInfo",
     "LoraRecipeInfo")

# ============================================================
# lora_dataset — validate + sha-pin a JSONL.
# ============================================================
def _lora_dataset_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".jsonl")
    sha_out = ctx.actions.declare_file(ctx.label.name + ".sha")
    args = ctx.actions.args()
    args.add("--in", ctx.file.src.path)
    args.add("--out", out.path)
    args.add("--sha-out", sha_out.path)
    args.add("--schema", ctx.attr.schema)
    args.add("--min-examples", ctx.attr.min_examples)
    ctx.actions.run(
        executable = ctx.executable._validator,
        inputs = [ctx.file.src],
        outputs = [out, sha_out],
        arguments = [args],
        mnemonic = "LoraDatasetValidate",
        progress_message = "Validating LoRA dataset %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([out, sha_out])),
        LoraDatasetInfo(
            jsonl = out,
            schema = ctx.attr.schema,
            # n_examples and sha are emitted into sha_out as a JSON
            # sidecar; build-time consumers read it via a separate
            # action when they need the actual values.
            n_examples = -1,  # placeholder — real value in the sidecar
            sha = "",         # placeholder — real value in the sidecar
        ),
    ]

lora_dataset = rule(
    implementation = _lora_dataset_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".jsonl"]),
        "schema": attr.string(
            default = "messages_v1",
            values = ["messages_v1", "instruction_v1"],
        ),
        "min_examples": attr.int(default = 1),
        "_validator": attr.label(
            default = "@rules_lora//runtime/torchtune_runner:validate_jsonl",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# lora_recipe — render a YAML config (torchtune/axolotl/peft).
# ============================================================
def _lora_recipe_impl(ctx):
    yaml = ctx.actions.declare_file(ctx.label.name + ".yaml")
    args = ctx.actions.args()
    args.add("--framework", ctx.attr.framework)
    args.add("--rank", ctx.attr.rank)
    args.add("--alpha", ctx.attr.alpha)
    args.add_joined("--target-modules", ctx.attr.target_modules, join_with = ",")
    args.add("--learning-rate", ctx.attr.learning_rate)
    args.add("--micro-batch-size", ctx.attr.micro_batch_size)
    args.add("--grad-accum-steps", ctx.attr.grad_accum_steps)
    args.add("--epochs", ctx.attr.epochs)
    args.add("--out", yaml.path)
    ctx.actions.run(
        executable = ctx.executable._renderer,
        outputs = [yaml],
        arguments = [args],
        mnemonic = "LoraRecipeRender",
        progress_message = "Rendering LoRA recipe %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([yaml])),
        LoraRecipeInfo(
            yaml = yaml,
            framework = ctx.attr.framework,
            rank = ctx.attr.rank,
            alpha = ctx.attr.alpha,
            target_modules = ctx.attr.target_modules,
            epochs = ctx.attr.epochs,
            sha = "",  # filled by the renderer's sidecar
        ),
    ]

lora_recipe = rule(
    implementation = _lora_recipe_impl,
    attrs = {
        "framework": attr.string(
            default = "torchtune",
            values = ["torchtune", "axolotl", "peft", "trl"],
        ),
        "rank": attr.int(default = 16),
        "alpha": attr.int(default = 32),
        "target_modules": attr.string_list(
            default = ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        "learning_rate": attr.string(default = "2e-4"),
        "micro_batch_size": attr.int(default = 4),
        "grad_accum_steps": attr.int(default = 8),
        "epochs": attr.int(default = 3),
        "_renderer": attr.label(
            default = "@rules_lora//runtime/torchtune_runner:render_recipe",
            executable = True,
            cfg = "exec",
        ),
    },
)

# ============================================================
# lora_base_model — pin an HF hub model by revision.
# ============================================================
def _lora_base_model_impl(ctx):
    return [
        DefaultInfo(),
        LoraBaseModelInfo(
            id = ctx.attr.repo,
            revision = ctx.attr.revision,
            config_path = ctx.file.config if ctx.file.config else None,
        ),
    ]

lora_base_model = rule(
    implementation = _lora_base_model_impl,
    attrs = {
        "repo": attr.string(mandatory = True, doc = "HF hub repo id."),
        "revision": attr.string(
            mandatory = True,
            doc = "Commit sha or `sha256:<digest>` of the model snapshot.",
        ),
        "config": attr.label(
            allow_single_file = [".json"],
            doc = "Optional local model_config.json override.",
        ),
    },
)

# ============================================================
# lora_train — orchestrate a training run.
# ============================================================
#
# `bazel run :<name>.train` (the wrapper macro emits a `_train`
# executable target). The rule itself produces the adapter file as
# a build output if the backend supports declarative outputs (local
# CPU smoke). For remote backends (runpod), the build produces a
# job-spec file and the executable does the run + download.
def _lora_train_impl(ctx):
    spec = ctx.actions.declare_file(ctx.label.name + ".jobspec.json")
    args = ctx.actions.args()
    args.add("write-jobspec")
    args.add("--name", ctx.label.name)
    args.add("--recipe", ctx.attr.recipe[LoraRecipeInfo].yaml.path)
    args.add("--dataset", ctx.attr.dataset[LoraDatasetInfo].jsonl.path)
    args.add("--base-id", ctx.attr.base[LoraBaseModelInfo].id)
    args.add("--base-revision", ctx.attr.base[LoraBaseModelInfo].revision)
    args.add("--backend", ctx.attr.backend)
    args.add("--out", spec.path)
    ctx.actions.run(
        executable = ctx.executable._spec_writer,
        inputs = [
            ctx.attr.recipe[LoraRecipeInfo].yaml,
            ctx.attr.dataset[LoraDatasetInfo].jsonl,
        ],
        outputs = [spec],
        arguments = [args],
        mnemonic = "LoraJobSpec",
        progress_message = "Composing LoRA job spec for %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([spec])),
    ]

lora_train = rule(
    implementation = _lora_train_impl,
    attrs = {
        "base": attr.label(
            mandatory = True,
            providers = [LoraBaseModelInfo],
        ),
        "recipe": attr.label(
            mandatory = True,
            providers = [LoraRecipeInfo],
        ),
        "dataset": attr.label(
            mandatory = True,
            providers = [LoraDatasetInfo],
        ),
        "backend": attr.string(
            default = "runpod",
            values = ["local", "runpod", "modal"],
        ),
        "_spec_writer": attr.label(
            default = "@rules_lora//runtime/runpod_orchestrator:write_jobspec",
            executable = True,
            cfg = "exec",
        ),
    },
)
