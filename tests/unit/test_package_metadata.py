"""What the package says about itself.

Everything here is metadata a user never sees until it is wrong on PyPI, at
which point it is wrong permanently: a release cannot be re-uploaded under the
same version. These are also the facts that live in several places at once
(pyproject, the package ``__init__``, the changelog, LICENSE) and so drift
silently: pyproject declared MIT while the repository had no licence file.
"""

import re
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from dietnet.cli import main

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "dietnet"

# Written out because a wrong slug in project.urls is a 404 on the PyPI sidebar
# and nothing in a checkout would notice.
SLUG = "MattScicluna/DIETNETWORK"
UPSTREAM_SLUG = "HussinLab/DIETNETWORK"


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((REPO / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def project(pyproject):
    return pyproject["project"]


class TestLicense:
    def test_the_declared_license_matches_the_license_file(self, project):
        first_line = (REPO / "LICENSE").read_text().splitlines()[0].strip()

        assert first_line.startswith("BSD 3-Clause")
        assert project["license"] == "BSD-3-Clause"

    def test_no_classifier_contradicts_the_license_file(self, project):
        license_classifiers = [c for c in project["classifiers"] if c.startswith("License ::")]

        assert not [c for c in license_classifiers if "BSD" not in c], license_classifiers

    def test_the_notice_excludes_the_upstream_code_and_ships(self, project):
        """The wheel redistributes HussinLab's code, which this licence cannot cover."""
        notice = (REPO / "NOTICE").read_text()

        assert "Dietnet_train_and_evaluate" in notice and UPSTREAM_SLUG in notice
        assert set(project["license-files"]) == {"LICENSE", "NOTICE"}


class TestVersion:
    def test_the_package_and_the_project_agree(self, project):
        declared = re.search(
            r'^__version__ = "([^"]+)"$', (PACKAGE / "__init__.py").read_text(), re.M
        )

        assert declared, "src/dietnet/__init__.py declares no __version__"
        assert declared.group(1) == project["version"]

    def test_it_is_a_release_version(self, project):
        # PEP 440, and no leftover dev/rc suffix on something about to be tagged.
        assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"]), project["version"]

    def test_the_command_line_reports_it(self, project):
        result = CliRunner().invoke(main, ["--version"])

        assert result.exit_code == 0 and project["version"] in result.output

    def test_it_is_written_down_in_exactly_one_source_file(self, project):
        """``__init__.py`` is the single place; everything else must read it."""
        elsewhere = [
            str(path.relative_to(REPO))
            for path in PACKAGE.rglob("*.py")
            if path.name != "__init__.py" and f'"{project["version"]}"' in path.read_text()
        ]

        assert elsewhere == []

    def test_the_changelog_documents_it(self, project):
        changelog = (REPO / "CHANGELOG.md").read_text()

        assert (
            f"## [{project['version']}]" in changelog
        ), f"CHANGELOG.md has no section for {project['version']}"


class TestTypeInformation:
    def test_the_package_ships_a_py_typed_marker(self):
        """PEP 561: a type checker ignores a dependency's inline types unless the
        distribution says they are there."""
        assert (PACKAGE / "py.typed").exists()


class TestUrls:
    def test_every_url_points_at_the_real_repository(self, project):
        wrong = {
            name: url
            for name, url in project["urls"].items()
            if "github.com" in url and SLUG not in url and UPSTREAM_SLUG not in url
        }

        assert not wrong, f"these do not point at github.com/{SLUG}: {wrong}"

    def test_it_names_a_homepage_documentation_and_an_issue_tracker(self, project):
        assert {"Homepage", "Documentation", "Issues"} <= set(project["urls"])


class TestDependencies:
    def test_the_toml_parser_the_tests_need_is_a_dev_dependency(self, pyproject):
        """This file imports it on 3.10; present transitively is not declared."""
        dev = pyproject["project"]["optional-dependencies"]["dev"]

        assert any(spec.startswith("tomli") for spec in dev), dev

    def test_no_runtime_dependency_is_unpinned_at_the_bottom(self, project):
        # A dependency with no lower bound resolves to whatever is on the index,
        # including versions predating the API this package uses.
        unbounded = [d for d in project["dependencies"] if not re.search(r"[><=~]", d)]

        assert not unbounded, unbounded

    def test_every_supported_python_is_a_classifier(self, project):
        """requires-python and the classifiers are the same claim, made twice."""
        low, high = re.fullmatch(r">=3\.(\d+),<3\.(\d+)", project["requires-python"]).groups()
        claimed = {
            c.rsplit(" ", 1)[1]
            for c in project["classifiers"]
            if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
        }

        assert claimed == {f"3.{minor}" for minor in range(int(low), int(high))}
