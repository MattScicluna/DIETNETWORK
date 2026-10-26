"""The README describes the command line that exists.

The README is where people learn the commands, so a command it does not
mention is one nobody finds, and one it mentions that was renamed sends
people to an error. This checks the mechanical half -- that the sets of
subcommands agree -- as manifold_genetics does for its docs. Prose still
needs a human.
"""

import re
from pathlib import Path

from dietnet.cli import main

README = Path(__file__).resolve().parents[2] / "README.md"


def documented() -> set:
    return set(re.findall(r"\bdietnet ([a-z][a-z0-9-]*)\b", README.read_text()))


def actual() -> set:
    return set(main.commands)


def test_every_command_is_in_the_readme():
    assert actual() - documented() == set(), "commands missing from README.md"


def test_the_readme_names_no_command_that_does_not_exist():
    assert documented() - actual() == set(), "README.md names commands that do not exist"
