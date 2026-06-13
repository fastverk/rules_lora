"""rules_lora backend toolchain.

The training backend (local / runpod / modal) is a **toolchain**, resolved
per-platform — not a per-target `backend = ...` attribute. This is the idiomatic
Bazel dispatch: register one `lora_backend_toolchain` per backend, gate each on a
`@rules_lora//lora/backend:backend` constraint, and let `lora_train` resolve the
one matching the build's platform.

    bazel build //... --platforms=@rules_lora//lora/backend:local_platform
    bazel build //...                                              # default: runpod

The toolchain carries the backend's *identity* + *tools* (the jobspec composer).
The per-backend `.run` entry is dispatched with the same constraint via `select()`
in `//lora:defs.bzl`, so the build side (this toolchain) and the run side stay in
lockstep on one platform choice.
"""

load(":providers.bzl", "LoraBackendInfo")

def _lora_backend_toolchain_impl(ctx):
    return [platform_common.ToolchainInfo(
        lora_backend = LoraBackendInfo(
            name = ctx.attr.backend_name,
            spec_writer = ctx.attr.spec_writer[DefaultInfo].files_to_run,
            default_image = ctx.attr.default_image,
            default_gpu = ctx.attr.default_gpu,
        ),
    )]

lora_backend_toolchain = rule(
    implementation = _lora_backend_toolchain_impl,
    attrs = {
        "backend_name": attr.string(
            mandatory = True,
            values = ["local", "runpod", "modal"],
            doc = "Backend identity baked into the jobspec.",
        ),
        "spec_writer": attr.label(
            mandatory = True,
            executable = True,
            cfg = "exec",
            doc = "The jobspec composer binary (e.g. //runtime/runpod_orchestrator:write_jobspec).",
        ),
        "default_image": attr.string(default = "", doc = "Default container image (remote backends)."),
        "default_gpu": attr.string(default = "", doc = "Default GPU type (remote backends)."),
    },
    doc = "Define a LoRA training backend as a toolchain. Register via `toolchain()` + `register_toolchains`.",
)
