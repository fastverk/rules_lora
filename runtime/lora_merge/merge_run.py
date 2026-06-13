#!/usr/bin/env python3
"""Adapter-merge entry point — `bazel run :<name>.run` for `lora_merge`.

A `py_binary`, not a generated shell wrapper: the //lora:defs.bzl macro emits a
`<name>.run` py_binary with this as `srcs`, the build-generated `<name>.merge.json`
config + the `lora-merge` rust binary in runfiles, and both runfiles paths passed
via `args` (`$(rlocationpath ...)`). Replaces the former generated bash wrapper,
so rules_lora emits no shell of its own.

This is a thin orchestrator: it resolves the candle-based `lora-merge` binary
(which does the actual fold) and invokes it via subprocess — a direct exec, no
shell interpretation. The adapter/out dirs are workspace-relative runtime paths
(training pulled the adapter to `outputs/`), so it runs from
`$BUILD_WORKSPACE_DIRECTORY`.

Config schema (`<name>.merge.json`, written by the lora_merge rule):
    {
      "adapter_dir": str, "base_id": str, "base_revision": str,
      "out_dir": str, "push_repo": str, "private": bool
    }
"""

import argparse
import json
import os
import subprocess
import sys

from python.runfiles import runfiles


def main():
    ap = argparse.ArgumentParser(description="LoRA adapter merge runner.")
    ap.add_argument("--config", required=True, help="rlocationpath of <name>.merge.json")
    ap.add_argument("--merge-tool", required=True, help="rlocationpath of the lora-merge binary")
    args, passthrough = ap.parse_known_args()

    r = runfiles.Create()
    config_path = r.Rlocation(args.config)
    merge_tool = r.Rlocation(args.merge_tool)
    if not config_path or not os.path.isfile(config_path):
        sys.exit("fatal: config not found via runfiles: %s" % args.config)
    if not merge_tool or not os.path.isfile(merge_tool):
        sys.exit("fatal: lora-merge not found via runfiles: %s" % args.merge_tool)

    with open(config_path) as fh:
        cfg = json.load(fh)

    cmd = [
        os.path.abspath(merge_tool),
        "--adapter", cfg["adapter_dir"],
        "--base-id", cfg["base_id"],
        "--base-revision", cfg["base_revision"],
        "--out", cfg["out_dir"],
    ]
    if cfg.get("push_repo"):
        cmd += ["--push-repo", cfg["push_repo"]]
        if cfg.get("private"):
            cmd.append("--private")
    cmd += passthrough

    # Resolve adapter_dir / out_dir against the source tree (where training
    # pulled the adapter to outputs/), not the runfiles sandbox.
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    os.chdir(workspace)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
