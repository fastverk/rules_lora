"""Render a LoRA training recipe as YAML.

Inputs come from the `lora_recipe` rule attrs. Output is a torchtune
/ axolotl / peft / trl YAML at `--out`. Framework-agnostic schema
inside this script; per-framework key mapping lives in
FRAMEWORK_KEYMAP.
"""

from __future__ import annotations

import argparse
import sys

# Per-framework key shape. Filled in lazily; only the active
# framework's keys get rendered.
TORCHTUNE_TEMPLATE = """\
# rules_lora-rendered torchtune recipe.
# DO NOT HAND-EDIT — re-render via `bazel build :<recipe_name>`.

# LoRA hyperparameters
lora:
  rank: {rank}
  alpha: {alpha}
  target_modules: {target_modules}

# Training loop
training:
  learning_rate: {lr}
  micro_batch_size: {mb}
  gradient_accumulation_steps: {gas}
  epochs: {epochs}

# Dataset + base model are bound at job-spec time by the orchestrator
# — they are not part of the recipe artifact (the recipe is reusable
# across datasets / bases).
"""

AXOLOTL_TEMPLATE = """\
# rules_lora-rendered axolotl recipe.
adapter: lora
lora_r: {rank}
lora_alpha: {alpha}
lora_target_modules:
{target_modules_yaml}
learning_rate: {lr}
micro_batch_size: {mb}
gradient_accumulation_steps: {gas}
num_epochs: {epochs}
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--framework", required=True,
                   choices=["torchtune", "axolotl", "peft", "trl"])
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=int, required=True)
    p.add_argument("--target-modules", dest="target_modules", required=True,
                   help="comma-separated list")
    p.add_argument("--learning-rate", dest="lr", required=True)
    p.add_argument("--micro-batch-size", dest="mb", type=int, required=True)
    p.add_argument("--grad-accum-steps", dest="gas", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    if args.framework == "torchtune":
        body = TORCHTUNE_TEMPLATE.format(
            rank=args.rank,
            alpha=args.alpha,
            target_modules=target_modules,
            lr=args.lr,
            mb=args.mb,
            gas=args.gas,
            epochs=args.epochs,
        )
    elif args.framework == "axolotl":
        body = AXOLOTL_TEMPLATE.format(
            rank=args.rank,
            alpha=args.alpha,
            target_modules_yaml="\n".join(f"  - {m}" for m in target_modules),
            lr=args.lr,
            mb=args.mb,
            gas=args.gas,
            epochs=args.epochs,
        )
    else:
        sys.exit(f"render_recipe: framework={args.framework!r} not yet supported "
                 f"(stub — add a template above to enable)")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"render_recipe: wrote {args.framework} YAML → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
