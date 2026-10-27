"""The Claude Code data-egress hook (.claude/hooks/egress_guard.py).

Cases are commands an agent working on this repository actually runs, including
the two cluster copies made on 2026-10-26 (a public 1KGP dataset and a UK
Biobank SNP list, both approved by a person, which is what ``ask`` gets).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "egress_guard.py"

spec = importlib.util.spec_from_file_location("egress_guard", HOOK)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "command",
    [
        "scp -q 'narval:$SCRATCH/dietnet_published/1KGP_dataset.hdf5' ~/dietnet/",
        "scp narval:projects/lab/UKB/ukb.bim ukbb/ukbb_array_qc.bim",
        "rsync -av --progress narval:runs/pipeline_ukbb/ ./ukbb_run/",
        "rsync -e 'ssh -J narval' ng30401:/tmp/out.tsv .",
        "cd /tmp && scp user@beluga.alliancecan.ca:~/predictions.tsv .",
        "sftp narval",
        "ssh narval 'cat $SCRATCH/cohort.bed' > cohort.bed",
        'ssh -o BatchMode=yes narval "zcat calls.vcf.gz" | gzip > calls.vcf.gz',
        "ssh narval 'scp $SCRATCH/cohort.bed elsewhere.example.org:'",
        "scp weights.pt collaborator.example.org:/tmp/",
        "rsync dataset.h5 user@myserver:/data/",
        "curl -T embedding.npz https://upload.example.org/",
        "curl -F 'file=@genotypes.bed' https://example.org/upload",
        "wget --post-file=cohort.fam https://example.org/",
        "caffeinate -i scp narval:a.npz .",
        "FOO=1 scp narval:a.tsv .",
    ],
)
def test_it_asks_before_data_can_leave(command):
    decision, reason = guard.decide(command)

    assert decision == "ask", reason


@pytest.mark.parametrize(
    "command",
    [
        "git commit --no-verify -m 'quick fix'",
        "git commit -nm 'quick fix'",
        "git push --no-verify origin feature",
        "cd repo && git commit -an -m wip",
    ],
)
def test_it_refuses_to_skip_the_commit_hooks(command):
    decision, _ = guard.decide(command)

    assert decision == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "scp reproduction/retrain.sbatch narval:dietnet_refactor/",
        "scp alphagenome_1kgp.tsv narval:features/",
        "rsync -av features.npz nc20120:/tmp/",
        "ssh narval 'squeue -u $USER'",
        "ssh narval 'wc -l cohort.bim; sacct -j 7418264 -X -n'",
        "ssh -o BatchMode=yes narval 'cd $SCRATCH && sbatch job.sbatch'",
        "git commit -m 'Normal commit'",
        "git commit --amend --no-edit",
        "git commit -am 'Fix the -n handling'",
        "git push -u origin feat/package-maturity",
        "uv run pytest -q tests/unit",
        "curl -s https://api.genome.ucsc.edu/getData/sequence?genome=hg38",
        "cat results.tsv | head",
        "ls *.bed",
    ],
)
def test_ordinary_work_passes(command):
    assert guard.decide(command) is None


def test_an_unparseable_transfer_is_asked_about():
    assert guard.decide("scp 'narval:a.bed .")[0] == "ask"


def test_the_hook_speaks_claude_codes_protocol():
    event = {"tool_name": "Bash", "tool_input": {"command": "scp narval:cohort.bed ."}}

    result = subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps(event), capture_output=True, text=True
    )

    out = json.loads(result.stdout)["hookSpecificOutput"]
    assert result.returncode == 0
    assert out["hookEventName"] == "PreToolUse" and out["permissionDecision"] == "ask"
    assert "cluster" in out["permissionDecisionReason"]


def test_other_tools_and_safe_commands_produce_no_output():
    for event in (
        {"tool_name": "Read", "tool_input": {"file_path": "/x/cohort.bed"}},
        {"tool_name": "Bash", "tool_input": {"command": "ls"}},
    ):
        result = subprocess.run(
            [sys.executable, str(HOOK)], input=json.dumps(event), capture_output=True, text=True
        )
        assert result.returncode == 0 and result.stdout == ""


def test_the_project_settings_run_the_hook_on_every_bash_command():
    settings = json.loads((HOOK.parents[1] / "settings.json").read_text())

    (entry,) = settings["hooks"]["PreToolUse"]
    assert entry["matcher"] == "Bash"
    assert entry["hooks"][0]["command"].endswith('.claude/hooks/egress_guard.py"')
