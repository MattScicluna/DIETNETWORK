#!/usr/bin/env python3
"""Claude Code PreToolUse hook: stop data leaving where it must stay.

Cohort data (UK Biobank, All of Us, CaG) stays on the cluster it was granted
on, and so does anything derived from it. The pre-commit hooks guard what
reaches git; this guards what an agent's shell commands move, which git never
sees. It reads the Bash command Claude Code is about to run and answers:

- ask:  any copy *from* a remote host to here (scp, rsync, sftp), whatever the
        file; the person decides each time. Also dumping a genotype or data
        file over ssh (``ssh host 'cat x.bed' > x.bed``), and sending one to a
        host that is not an Alliance cluster (scp/rsync upload, curl, wget).
- deny: ``git commit``/``git push`` with ``--no-verify``, which skips the
        sensitive-files and gitleaks hooks.

Everything else passes. "Sensitive" means the commit check's own rules
(scripts/check_sensitive_files.py: genotype and data-container suffixes), so
there is one definition. Stdlib only; reads the hook JSON on stdin and writes a
decision on stdout. Wired up in .claude/settings.json.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.check_sensitive_files import (  # noqa: E402
    DATA_CONTAINER_SUFFIXES,
    GENOTYPE_SUFFIXES,
)

SENSITIVE_SUFFIXES = GENOTYPE_SUFFIXES | DATA_CONTAINER_SUFFIXES | {".bcf", ".sample"}

# Alliance (Compute Canada) clusters and their compute nodes: sending data *to*
# these is where it is meant to live.
CLUSTER_HOST_RE = re.compile(
    r"^((narval|beluga|cedar|graham|niagara|fir|nibi|rorqual|trillium|killarney|vulcan)\d*"
    r"|(nc|ng|nl|bc|bg|bl|cdr|gra|nia)\d{3,5}"
    r"|.*\.(alliancecan|computecanada|calculcanada)\.ca)$"
)

COPY_TOOLS = {"scp", "rsync", "sftp"}
DUMP_TOOLS = {"cat", "zcat", "gzip", "head", "tail", "less", "more", "base64", "xxd", "od", "dd",
              "tar", "plink", "plink2", "bcftools"}  # fmt: skip
SEPARATORS = {";", "&&", "||", "|", "&", "\n", "(", ")"}

# scp's and rsync's options that take a value: the value is not an operand.
VALUE_OPTIONS = {
    "scp": {"-c", "-F", "-i", "-J", "-l", "-o", "-P", "-S"},
    "rsync": {"-e", "--rsh", "--exclude", "--include", "--filter", "-f", "--files-from",
              "--rsync-path", "--log-file", "--port", "--password-file"},  # fmt: skip
    "sftp": {"-b", "-c", "-F", "-i", "-J", "-l", "-o", "-P", "-R", "-S", "-s"},
}


def _sensitive(word: str) -> bool:
    path = word.rstrip("/'\"").split(":")[-1]
    return any(path.lower().endswith(suffix) for suffix in SENSITIVE_SUFFIXES)


def _host(operand: str) -> str | None:
    """The host of a remote operand (``[user@]host:path``, ``scp://...``), else None."""
    if operand.startswith(("scp://", "sftp://", "rsync://")):
        operand = operand.split("://", 1)[1].split("/", 1)[0]
        return operand.split("@")[-1].split(":")[0]
    if operand.startswith(("/", "./", "../", "~")) or ":" not in operand:
        return None
    host = operand.split(":", 1)[0]
    if "/" in host or not host:
        return None
    return host.split("@")[-1]


def _is_cluster(host: str) -> bool:
    return bool(CLUSTER_HOST_RE.match(host))


def _segments(command: str) -> list[list[str]]:
    """Simple commands of a shell line, as word lists (operators split them)."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments, current = [], []
    for token in lexer:
        if token in SEPARATORS or set(token) <= set(";&|()"):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _operands(tool: str, args: list[str]) -> list[str]:
    operands, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg in VALUE_OPTIONS.get(tool, ()):
            skip = True
        elif not arg.startswith("-") or arg == "-":
            operands.append(arg)
    return operands


def _check_copy(tool: str, args: list[str]) -> str | None:
    operands = _operands(tool, args)
    if tool == "sftp":
        hosts = [_host(op) or op.split("@")[-1] for op in operands[:1]]
        return f"sftp session to {hosts[0]}: files can be fetched from it" if hosts else None
    if len(operands) < 2:
        return None
    *sources, destination = operands
    remote_sources = [s for s in sources if _host(s)]
    if remote_sources and not _host(destination):
        return f"{tool} copies {', '.join(remote_sources)} from a remote host to this machine"
    dest_host = _host(destination)
    if dest_host and not _is_cluster(dest_host) and any(_sensitive(s) for s in sources):
        return f"{tool} sends a genotype or data file to {dest_host}, which is not a cluster"
    return None


def _check_ssh(args: list[str]) -> str | None:
    """``ssh host 'cat x.bed'``: the file's bytes come back over the connection."""
    words, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg in {"-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m",
                     "-O", "-o", "-p", "-Q", "-R", "-S", "-W", "-w"}:  # fmt: skip
            skip = True
        elif not arg.startswith("-"):
            words.append(arg)
    if len(words) < 2:
        return None
    host, remote = words[0].split("@")[-1], " ".join(words[1:])
    for segment in _segments_or_none(remote) or []:
        tool = Path(segment[0]).name
        if tool in DUMP_TOOLS and any(_sensitive(w) for w in segment[1:]):
            return f"ssh {host} runs {tool} on a genotype or data file; its contents come back here"
        reason = _check_segment(segment)
        if reason:
            return f"on {host}: {reason}"
    return None


def _check_upload(tool: str, args: list[str]) -> str | None:
    if tool == "curl":
        for i, arg in enumerate(args):
            value = args[i + 1] if i + 1 < len(args) else ""
            if arg in {"-T", "--upload-file"} and _sensitive(value):
                return f"curl uploads {value}"
            if arg in {"-F", "--form", "-d", "--data", "--data-binary"} and "@" in value:
                if _sensitive(value.split("@", 1)[1].split(";")[0]):
                    return f"curl posts the file in {value!r}"
    if tool == "wget":
        for arg in args:
            if arg.startswith(("--post-file=", "--body-file=")) and _sensitive(
                arg.split("=", 1)[1]
            ):
                return f"wget posts {arg.split('=', 1)[1]}"
    return None


def _check_git(args: list[str]) -> tuple[str, str] | None:
    if args[:1] in (["commit"], ["push"]) and (
        "--no-verify" in args
        or (args[0] == "commit" and any(re.fullmatch(r"-\w*n\w*", a) for a in args[1:]))
    ):
        return ("deny", f"git {args[0]} --no-verify skips the sensitive-files and gitleaks hooks")
    return None


def _check_segment(segment: list[str]) -> str | None:
    tool, args = Path(segment[0]).name, segment[1:]
    if tool in COPY_TOOLS:
        return _check_copy(tool, args)
    if tool == "ssh":
        return _check_ssh(args)
    return _check_upload(tool, args)


def _segments_or_none(command: str) -> list[list[str]] | None:
    try:
        return _segments(command)
    except ValueError:  # unbalanced quotes: not a command the shell would run either
        return None


def decide(command: str) -> tuple[str, str] | None:
    """("ask" | "deny", reason) for a command that could move data out, else None."""
    segments = _segments_or_none(command)
    if segments is None:
        if re.search(r"\b(scp|rsync|sftp|ssh|curl|wget)\b", command):
            return ("ask", "could not parse a command that may transfer files")
        return None
    for segment in segments:
        while segment and re.fullmatch(r"\w+=.*", segment[0]):  # VAR=value prefixes
            segment = segment[1:]
        if segment and Path(segment[0]).name in {"sudo", "env", "nohup", "time", "caffeinate"}:
            segment = [w for w in segment[1:] if not w.startswith("-")]
        if not segment:
            continue
        if Path(segment[0]).name == "git":
            verdict = _check_git(segment[1:])
            if verdict:
                return verdict
            continue
        reason = _check_segment(segment)
        if reason:
            return ("ask", reason)
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if event.get("tool_name") != "Bash":
        return 0
    verdict = decide(event.get("tool_input", {}).get("command", ""))
    if verdict is None:
        return 0
    decision, reason = verdict
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": (
                        f"Data-egress guard: {reason}. Cohort data and anything derived "
                        "from it stays on the cluster (see CLAUDE.md)."
                    ),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
