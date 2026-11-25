"""Edge cases of dietnet.upstream: finding upstream's code, the manifest check and failing
scripts."""

from __future__ import annotations

import hashlib

import pytest

from dietnet import upstream


def _train_dir(base):
    """``base`` with an (empty) upstream train.py in it."""
    (base / upstream.TRAIN_DIR).mkdir(parents=True)
    (base / upstream.TRAIN_DIR / "train.py").write_text("")
    return base


def test_root_prefers_an_installed_copy_then_the_checkout(tmp_path, monkeypatch):
    """A wheel's _upstream/ wins over the source tree; with neither, the error says
    where it looked."""
    package = tmp_path / "src" / "dietnet"
    package.mkdir(parents=True)
    monkeypatch.setattr(upstream, "_PACKAGE", package)
    find = upstream.root.__wrapped__  # the uncached function

    with pytest.raises(FileNotFoundError, match="Dietnet_train_and_evaluate/ not found next to"):
        find()

    _train_dir(tmp_path)  # a source checkout: two levels above the package
    assert find() == tmp_path
    _train_dir(package / "_upstream")  # an installed wheel
    assert find() == package / "_upstream"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_verify_reports_missing_modified_and_added_files(tmp_path, monkeypatch):
    """Each kind of difference against the manifest is reported, by path; compiled
    caches are not upstream's files and are ignored."""
    kept = _write(tmp_path / "code" / "kept.py", "same")
    _write(tmp_path / "code" / "edited.py", "changed")
    _write(tmp_path / "code" / "sub" / "added.py", "new")
    _write(tmp_path / "code" / "__pycache__" / "kept.cpython-310.pyc", "bytes")
    files = {
        "code/kept.py": kept,
        "code/edited.py": hashlib.sha256(b"original").hexdigest(),
        "code/removed.py": hashlib.sha256(b"gone").hexdigest(),
    }
    monkeypatch.setattr(upstream, "root", lambda: tmp_path)
    monkeypatch.setattr(upstream, "manifest", lambda: {"files": files, "directories": ["code"]})

    assert upstream.verify() == [
        "modified: code/edited.py",
        "missing: code/removed.py",
        "added: code/sub/added.py",
    ]

    (tmp_path / "code" / "sub" / "added.py").unlink()
    _write(tmp_path / "code" / "edited.py", "original")
    _write(tmp_path / "code" / "removed.py", "gone")
    assert upstream.verify() == []


FAILING = """\
import os, sys
print("args", sys.argv[1:])
print("cwd", os.getcwd())
print("hashseed", os.environ["PYTHONHASHSEED"], "extra", os.environ.get("DIETNET_TEST_EXTRA"))
for i in range(40):
    print("line", i)
sys.exit(3)
"""


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    """Upstream scripts replaced by ones in ``tmp_path``."""
    (tmp_path / "fail.py").write_text(FAILING)
    (tmp_path / "ok.py").write_text("print('fine')\n")
    monkeypatch.setattr(upstream, "script", lambda name: tmp_path / name)
    return tmp_path


def test_a_failing_script_with_a_log_raises_with_its_tail(scripts):
    """The error gives the script, status and log path, and the log's last 25 lines;
    the script ran from its own directory with the extra environment and a fixed hash
    seed."""
    log = scripts / "logs" / "fail.log"
    with pytest.raises(upstream.UpstreamError, match=f"fail.py exited with status 3; log {log}:"):
        upstream.run("fail.py", ["--flag", 7], log=log, env={"DIETNET_TEST_EXTRA": "yes"})
    text = log.read_text()
    assert "args ['--flag', '7']" in text
    assert f"cwd {scripts.resolve()}" in text or f"cwd {scripts}" in text
    assert "hashseed 0 extra yes" in text

    with pytest.raises(upstream.UpstreamError) as caught:
        upstream.run("fail.py", [], log=log)
    tail = str(caught.value).split(":\n", 1)[1]
    assert tail.splitlines() == [f"line {i}" for i in range(15, 40)]


def test_a_failing_script_without_a_log_raises_with_its_status(scripts, capfd):
    """Without a log, output goes to this process's and the error gives the status only."""
    with pytest.raises(upstream.UpstreamError, match=r"^fail\.py exited with status 3$"):
        upstream.run("fail.py", [])
    assert "line 39" in capfd.readouterr().out
    assert upstream.run("ok.py", []) is None
