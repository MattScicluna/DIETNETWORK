"""The README and the documentation describe the command line that exists.

The README is where people learn the commands, and docs/cli.md is the
reference, so a command they do not mention is one nobody finds, and one they
mention that was renamed sends people to an error. This checks the mechanical
half -- that the sets of subcommands agree, and that docs/cli.md gives every
command a section naming all of its options and no others -- as
manifold_genetics does for its docs. Prose still needs a human.
"""

import re
from pathlib import Path

import pytest

from dietnet.cli import main

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
CLI_DOC = ROOT / "docs" / "cli.md"


def documented(path: Path) -> set:
    return set(re.findall(r"\bdietnet ([a-z][a-z0-9-]*)\b", path.read_text()))


def actual() -> set:
    return set(main.commands)


@pytest.mark.parametrize("doc", [README, CLI_DOC], ids=lambda p: p.name)
def test_every_command_is_documented(doc):
    assert actual() - documented(doc) == set(), f"commands missing from {doc.name}"


@pytest.mark.parametrize("doc", [README, CLI_DOC], ids=lambda p: p.name)
def test_the_docs_name_no_command_that_does_not_exist(doc):
    assert documented(doc) - actual() == set(), f"{doc.name} names commands that do not exist"


def sections() -> dict:
    """docs/cli.md's ``## `dietnet <command>` `` sections, by command."""
    parts = re.split(r"^## `dietnet ([a-z0-9-]+)`$", CLI_DOC.read_text(), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def test_every_command_has_a_section_in_cli_md():
    assert set(sections()) == actual()


@pytest.mark.parametrize("command", sorted(main.commands))
def test_cli_md_names_exactly_the_options_of_each_command(command):
    options = {
        name
        for param in main.commands[command].params
        for name in (*param.opts, *param.secondary_opts)
        if name.startswith("--")
    }
    named = set(re.findall(r"`(--[a-z][a-z0-9-]*)", sections()[command]))
    named |= set(re.findall(r" / (--[a-z][a-z0-9-]*)", sections()[command]))
    assert options - named == set(), f"options of {command} missing from docs/cli.md"
    assert named - options == set(), f"docs/cli.md gives {command} options it does not have"
