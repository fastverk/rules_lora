"""rules_lora public API.

```starlark
load("@rules_lora//lora:defs.bzl",
     "lora_base_model",
     "lora_dataset",
     "lora_recipe",
     "lora_train",
     "expert_manifest")
```

All four macros forward to rules in `//lora/private:rules.bzl`,
which in turn use providers from `//lora/private:providers.bzl`.

The macros are thin wrappers (no logic) — keep them stable; let
the underlying rules evolve.
"""

load(
    "//lora/private:rules.bzl",
    _lora_base_model = "lora_base_model",
    _lora_dataset = "lora_dataset",
    _lora_recipe = "lora_recipe",
    _lora_train = "lora_train",
)

# Re-exports — public surface.
lora_base_model = _lora_base_model
lora_dataset = _lora_dataset
lora_recipe = _lora_recipe
lora_train = _lora_train

def expert_manifest(name, adapters, routing = "nearest_centroid",
                    cluster_manifest = None, visibility = None):
    """Bundle N trained adapters into an `ExpertManifest.binpb`.

    The binpb wire-shape matches
    `[[rules_agentic_ide]]/proto/agentic_ide/v1/experts.proto`
    so the router in that repo can consume the output directly.

    Args:
      name: target name.
      adapters: list of labels to `lora_train` targets.
      routing: routing policy. One of {nearest_centroid,
               hull_membership, soft_top_k}.
      cluster_manifest: optional label to a `ClusterManifest.binpb`
                        that this expert set is paired with.
      visibility: standard bazel visibility.

    TODO(v0.2): emit the binpb action. v0.1 stops at the JSON
    sidecar shape, which is enough for the routing prototype.
    """

    # Placeholder filegroup; real rule lands in v0.2 once we have
    # at least one `lora_train` target producing a real adapter.
    native.filegroup(
        name = name,
        srcs = adapters,
        visibility = visibility,
    )
    _ = routing
    _ = cluster_manifest
