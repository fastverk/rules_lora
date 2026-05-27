"""Validate an SFT JSONL.

Reads --in, verifies each line is JSON, optionally checks schema,
writes --out (re-emitted JSONL — currently a verbatim copy; the
re-emit gives us a place to do later normalizations) and --sha-out
(a small JSON sidecar with `{"sha": "...", "n_examples": N}`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMAS = {
    "messages_v1": {
        "required_top_level": ["messages"],
        "messages_required_per_item": ["role", "content"],
        "allowed_roles": {"system", "user", "assistant", "tool"},
    },
    "instruction_v1": {
        "required_top_level": ["instruction", "output"],
    },
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sha-out", dest="sha_out", required=True)
    p.add_argument("--schema", default="messages_v1", choices=list(SCHEMAS))
    p.add_argument("--min-examples", type=int, default=1)
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    sha_out = Path(args.sha_out)
    schema = SCHEMAS[args.schema]

    h = hashlib.blake2b(digest_size=32)
    n = 0
    with src.open("rb") as f, out.open("wb") as o:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                sys.exit(f"validate_jsonl: line {n+1} is not valid JSON: {e}")
            _validate(row, args.schema, schema, n + 1)
            h.update(raw_line)
            o.write(raw_line)
            if not raw_line.endswith(b"\n"):
                o.write(b"\n")
            n += 1

    if n < args.min_examples:
        sys.exit(f"validate_jsonl: only {n} examples, --min-examples={args.min_examples}")

    sha = h.hexdigest()
    sha_out.write_text(json.dumps({
        "sha": sha,
        "n_examples": n,
        "schema": args.schema,
    }) + "\n")
    print(f"validate_jsonl: {n} examples, sha={sha[:16]}…, schema={args.schema}", file=sys.stderr)
    return 0


def _validate(row, schema_name: str, schema: dict, line_no: int) -> None:
    for k in schema["required_top_level"]:
        if k not in row:
            sys.exit(f"validate_jsonl: line {line_no} missing required field `{k}` for schema {schema_name}")
    if schema_name == "messages_v1":
        msgs = row["messages"]
        if not isinstance(msgs, list) or not msgs:
            sys.exit(f"validate_jsonl: line {line_no} `messages` must be a non-empty list")
        for i, m in enumerate(msgs):
            for k in schema["messages_required_per_item"]:
                if k not in m:
                    sys.exit(f"validate_jsonl: line {line_no} messages[{i}] missing `{k}`")
            if m["role"] not in schema["allowed_roles"]:
                sys.exit(f"validate_jsonl: line {line_no} messages[{i}].role={m['role']!r} not in {schema['allowed_roles']}")


if __name__ == "__main__":
    sys.exit(main())
