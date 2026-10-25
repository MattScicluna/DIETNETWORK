#!/usr/bin/env python3
"""Flag files that should not leave a machine holding controlled-access data.

Ported from manifold_genetics, where it was written, with one addition for this
repository: trained weights, genotype caches and HDF5 datasets (see
DATA_CONTAINER_SUFFIXES). DietNetwork is trained on the same clusters, next to
the same cohorts, and its model packages are built from them.

It exists there because of two incidents, both on a repository that is public:

* A ``git add -A examples/`` swept seven untracked files into a pull request,
  one of them a script whose own header says "NOT for external users".
* The 0.2.0 source distribution, built from the working tree, contained
  thirteen untracked files including agent tool state with cluster paths in it.

Neither was caught by review. The conclusion is not that more care is needed --
it is that a working directory on a cluster holding UK Biobank and All of Us
data is not a safe thing to bulk-add from, and that the check must be mechanical
and run before the commit rather than after the push.

Run it over specific paths (what the pre-commit hook does)::

    python -m scripts.check_sensitive_files path/to/file ...

or over everything git is tracking or about to track::

    python -m scripts.check_sensitive_files --staged
    python -m scripts.check_sensitive_files --tracked

Every rule below corresponds to something that leaked or nearly did. To allow a
specific path, add it to ``.sensitive-allow`` with a comment saying why: an
override should leave a reviewable trace in the repository, not be a flag
somebody types once and forgets.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ALLOWLIST = ".sensitive-allow"

# Phrases people write when they mean it. Matched case-insensitively, and only
# in a file's header: a notice like this is written at the top, so the same words
# further down are prose about the notice rather than the notice itself. Running
# this over the repository found three hits and all three were documentation of
# the leak it prevents -- a check whose only findings are itself gets switched
# off, so both that and the quoted-phrase case below are excluded.
HEADER_LINES = 40
PRIVATE_MARKERS = (
    "not for external users",
    "private script",
    "do not distribute",
    "do not share",
    "internal use only",
    "confidential",
)

# Genotype and phenotype containers. None of these belongs in a git history,
# whatever the cohort: even a public one is hundreds of megabytes.
GENOTYPE_SUFFIXES = frozenset(
    {".bed", ".bim", ".fam", ".ped", ".map", ".bgen", ".pgen", ".psam", ".pvar", ".vcf", ".gz"}
)

# What this repository's pipeline writes: HDF5 datasets, .npy genotype caches,
# .npz embeddings and statistics, .pt weights. Built from cohort genotypes, and
# often large; model packages are shared by download link, not through git.
DATA_CONTAINER_SUFFIXES = frozenset({".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth"})

# Directories holding local tool and agent state. These carry absolute paths,
# session identifiers and command history.
LOCAL_STATE_PREFIXES = (".claude/", ".superpowers/", ".vscode/", ".idea/")

PRIVATE_NAME_RE = re.compile(r"(^|[._-])(private|secret|credential|token|password)", re.I)

# Absolute paths that only exist on one cluster. They identify the machine, the
# allocation, and often the user.
CLUSTER_PATH_RE = re.compile(r"/(lustre\w*|scratch|project/(ctb|def|rrg)-)[/\w-]*")

# A biobank application identifier: an all-numeric id of 6-9 digits. Deliberately
# not matched against public cohort ids (HG00096, NA12878), which appear all over
# the examples and are meant to.
BIOBANK_ID_RE = re.compile(r"^\d{6,9}$")
MIN_ID_ROWS = 50
ID_ROW_FRACTION = 0.9

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".sh",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
        ".cfg",
        ".toml",
        ".ipynb",
    }
)


@dataclass(frozen=True)
class Finding:
    """One reason one file should not be committed."""

    path: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.rule}] {self.detail}"


def load_allowlist(repo_root: Path) -> frozenset:
    path = repo_root / ALLOWLIST
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _read_text(path: Path) -> Optional[str]:
    """The file's text, or None if it is not text we can scan."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _looks_like_an_identifier_column(text: str) -> Optional[str]:
    """Whether the first column is a long list of biobank identifiers."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < MIN_ID_ROWS:
        return None

    # Skip a header row if the first field is not itself an identifier.
    rows = lines[1:] if not BIOBANK_ID_RE.match(lines[0].split(",")[0].strip()) else lines
    if len(rows) < MIN_ID_ROWS:
        return None

    first = [row.split(",")[0].strip() for row in rows]
    matches = sum(1 for value in first if BIOBANK_ID_RE.match(value))
    if matches / len(first) < ID_ROW_FRACTION:
        return None
    return f"{matches} of {len(first)} rows begin with a 6-9 digit identifier"


def _private_notice(text: str) -> Optional[str]:
    """The header line declaring this file private, if there is one.

    Only the first :data:`HEADER_LINES` lines are considered, and a line that
    merely quotes the phrase is not a declaration -- that is how documentation
    of this check avoids tripping it.
    """
    for line in text.splitlines()[:HEADER_LINES]:
        lowered = line.lower()
        for marker in PRIVATE_MARKERS:
            position = lowered.find(marker)
            if position < 0:
                continue
            if _inside_a_quotation(line, position):
                continue  # prose about the phrase, not the phrase
            return line.strip()
    return None


def _inside_a_quotation(line: str, position: int) -> bool:
    """Whether ``position`` falls inside a quoted span of ``line``.

    Counting the quote characters before it, rather than looking for one
    immediately before, because prose quotes a whole header rather than a
    phrase: `a script headed "Private script ... NOT for external users"`.
    """
    return any(line.count(quote, 0, position) % 2 == 1 for quote in ('"', "'", "`"))


def check_path(path: Path, *, repo_root: Path) -> List[Finding]:
    """Every reason ``path`` should not be committed."""
    if not path.is_file():
        return []

    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    findings: List[Finding] = []

    def flag(rule: str, detail: str) -> None:
        findings.append(Finding(path=relative, rule=rule, detail=detail))

    if any(relative.startswith(prefix) for prefix in LOCAL_STATE_PREFIXES):
        flag("local-state", "local tool or agent state; carries absolute paths and history")

    if path.suffix.lower() in GENOTYPE_SUFFIXES:
        flag("genotype-data", f"{path.suffix} is a genotype or archive container")

    if path.suffix.lower() in DATA_CONTAINER_SUFFIXES:
        flag("data-container", f"{path.suffix} holds data or weights derived from genotypes")

    if PRIVATE_NAME_RE.search(path.name):
        flag("private-name", f"the filename {path.name!r} says it is private")

    text = _read_text(path)
    if text is not None:
        notice = _private_notice(text)
        if notice:
            flag("private-marker", f"says {notice!r}")

        cluster = CLUSTER_PATH_RE.search(text)
        if cluster:
            flag("cluster-path", f"contains the absolute path {cluster.group(0)!r}")

        identifiers = _looks_like_an_identifier_column(text)
        if identifiers:
            flag("sample-identifiers", identifiers)

    return findings


def check_paths(paths: Iterable[Path], *, repo_root: Path) -> List[Finding]:
    """Check several paths, dropping anything the allowlist permits.

    An allowlist entry is either a path, which waives every rule on it, or
    ``path:rule``, which waives one. Prefer the scoped form: waiving a file
    outright means a later edit that adds something genuinely sensitive to it
    goes unnoticed.
    """
    allowed = load_allowlist(repo_root)
    findings: List[Finding] = []
    for path in paths:
        for finding in check_path(Path(path), repo_root=repo_root):
            if finding.path in allowed or f"{finding.path}:{finding.rule}" in allowed:
                continue
            findings.append(finding)
    return findings


def _git(repo_root: Path, *args: str) -> List[str]:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--staged", action="store_true", help="check everything staged for commit")
    parser.add_argument("--tracked", action="store_true", help="check everything git tracks")
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="repository root (default: cwd)"
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    paths = list(args.paths)
    if args.staged:
        paths += [
            repo_root / name
            for name in _git(repo_root, "diff", "--cached", "--name-only", "--diff-filter=ACM")
        ]
    if args.tracked:
        paths += [repo_root / name for name in _git(repo_root, "ls-files")]

    if not paths:
        parser.error("give paths, or --staged, or --tracked")

    findings = check_paths(paths, repo_root=repo_root)
    if not findings:
        return 0

    print("Refusing to proceed: these files look like they should not be shared.\n")
    for finding in findings:
        print(f"  {finding}")
    print(
        f"\nIf one of these is genuinely safe, add it to {ALLOWLIST} with a comment"
        "\nsaying why, so the decision is reviewable. Prefer scoping the exemption"
        "\nto the one rule -- 'path/to/file:rule-name' -- rather than the whole file."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
