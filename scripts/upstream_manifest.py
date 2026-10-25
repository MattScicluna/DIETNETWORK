#!/usr/bin/env python3
"""Record which HussinLab/DIETNETWORK code this repository runs, file by file.

The training and model code in Dietnet_train_and_evaluate/ is HussinLab's.
This repository keeps only the files the ``dietnet`` API runs -- the four
scripts it calls and what they import -- and removes the rest (deprecated code,
experiments, upstream's own inference scripts, which ``dietnet predict``
replaces and tests/parity checks against). Every kept file must be byte for
byte the upstream commit's.

src/dietnet/upstream_manifest.json records the commit, the SHA-256 of each kept
file and the list of removed ones; tests/unit/test_upstream_unchanged.py fails
if a kept file differs or anything else appears.

After merging a newer upstream main (``git merge upstream/main``; delete again
any removed file the merge brings back), regenerate and review:

    python scripts/upstream_manifest.py --commit upstream/main
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIRS = ("Dietnet_train_and_evaluate", "Dietnet_inference")
MANIFEST = ROOT / "src" / "dietnet" / "upstream_manifest.json"


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", required=True, help="upstream commit (sha or ref)")
    args = parser.parse_args()

    sha = git("rev-parse", f"{args.commit}^{{commit}}").decode().strip()
    kept, removed, modified = {}, [], []
    for directory in UPSTREAM_DIRS:
        names = git("ls-tree", "-r", "--name-only", sha, "--", directory).decode().splitlines()
        for name in names:
            path = ROOT / name
            if not path.is_file():
                removed.append(name)
                continue
            digest = hashlib.sha256(git("show", f"{sha}:{name}")).hexdigest()
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                modified.append(name)
            kept[name] = digest

    if modified:
        sys.exit("refusing to record modified upstream files: " + ", ".join(modified))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "repository": "https://github.com/HussinLab/DIETNETWORK",
                "commit": sha,
                "directories": [d for d in UPSTREAM_DIRS if (ROOT / d).is_dir()],
                "files": dict(sorted(kept.items())),
                "removed": sorted(removed),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(kept)} kept, {len(removed)} removed, from {sha}")


if __name__ == "__main__":
    main()
