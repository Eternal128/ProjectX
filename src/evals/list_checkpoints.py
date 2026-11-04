"""Check configs/models.yaml checkpoints against the branches on Hugging Face.

    python -m evals.list_checkpoints --config configs/models.yaml
"""
from __future__ import annotations

import argparse
import sys

from common.config import load_config


def available_steps(hf_id: str) -> set[int]:
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(hf_id)
    return {int(b.name[4:]) for b in refs.branches if b.name.startswith("step") and b.name[4:].isdigit()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/models.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    wanted = set(cfg["checkpoints"])
    ok = True
    for m in cfg["models"]:
        have = available_steps(m["hf_id"])
        missing = sorted(wanted - have)
        ok &= not missing
        print(f"{m['hf_id']}: {len(have)} step branches, missing from config list: {missing or 'none'}")
    print("All configured checkpoints exist." if ok else "Some checkpoints are missing; fix configs/models.yaml.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
