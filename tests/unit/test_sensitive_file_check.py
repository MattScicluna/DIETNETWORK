"""The check that stands between a working directory and a public repository.

Ported with scripts/check_sensitive_files.py from manifold_genetics, where the
incidents below happened; the data-container tests at the end are this
repository's.

Written because it has already happened here twice. A `git add -A examples/`
swept seven untracked files into a pull request on a public repo, one of them a
script headed "Private script ... NOT for external users". Then the 0.2.0 source
distribution, built from the working tree, contained thirteen untracked files
including local tool state with cluster paths in it.

Neither was caught by review; both were caught by accident. The lesson is not
"be careful" -- it is that a working directory on a machine with controlled-access
data is not a safe thing to bulk-add from, and that the check has to be
mechanical.

These tests are the specification of what counts as sensitive. Every rule here
corresponds to something that actually leaked or nearly did.
"""

from scripts.check_sensitive_files import Finding, check_paths

MARKER_SCRIPT = """#!/bin/bash
#
# Private script to create UKBB sample lists and labels using internal metadata
# This is NOT for external users - they should create sample lists manually
#
set -e
"""


def rules(findings):
    return {f.rule for f in findings}


class TestPrivateMarkers:
    def test_a_file_that_says_it_is_not_for_external_users_is_flagged(self, tmp_path):
        path = tmp_path / "prepare_data_create_labels.sh"
        path.write_text(MARKER_SCRIPT)

        assert "private-marker" in rules(check_paths([path], repo_root=tmp_path))

    def test_the_finding_quotes_the_line_that_triggered_it(self, tmp_path):
        # A finding nobody can act on gets ignored, and then overridden. Which of
        # the file's two notices is quoted does not matter; that it quotes a real
        # line from the file does.
        path = tmp_path / "s.sh"
        path.write_text(MARKER_SCRIPT)

        (finding,) = [
            f for f in check_paths([path], repo_root=tmp_path) if f.rule == "private-marker"
        ]

        quoted = finding.detail.removeprefix("says ").strip("'\"")
        assert quoted in [line.strip() for line in MARKER_SCRIPT.splitlines()]

    def test_prose_quoting_the_phrase_is_not_flagged(self, tmp_path):
        """Documenting the rule must not trip the rule.

        Running this over the repository found exactly three hits, and all three
        were files describing the leak it exists to prevent -- the changelog, the
        release guide, and the test that checks the sdist. A check whose only
        findings are documentation of itself gets switched off.
        """
        path = tmp_path / "releasing.md"
        path.write_text(
            "The sdist carried a script whose header says "
            '"NOT for external users", which is why this is checked.\n'
        )

        assert check_paths([path], repo_root=tmp_path) == []

    def test_a_marker_inside_a_longer_quotation_is_not_flagged(self, tmp_path):
        """The quote can open well before the phrase.

        Prose about this check tends to quote a whole header rather than a
        phrase, so checking only whether a quote sits immediately before the
        marker is not enough.
        """
        path = tmp_path / "notes.md"
        path.write_text(
            'a script headed "Private script ... NOT for external users". Then it shipped.\n'
        )

        assert check_paths([path], repo_root=tmp_path) == []

    def test_the_notice_must_be_in_the_file_header(self, tmp_path):
        # A notice like this is written at the top of a file. The same words
        # three hundred lines into a document are prose about it.
        path = tmp_path / "long.md"
        path.write_text("filler\n" * 100 + "this is NOT for external users\n")

        assert check_paths([path], repo_root=tmp_path) == []

    def test_an_ordinary_script_is_not_flagged(self, tmp_path):
        path = tmp_path / "prepare_data.sh"
        path.write_text("#!/bin/bash\n# Build the fit and project subsets.\nset -e\n")

        assert check_paths([path], repo_root=tmp_path) == []


class TestGenotypeData:
    def test_plink_files_are_flagged(self, tmp_path):
        for name in ("x.bed", "x.bim", "x.fam", "x.pgen", "x.psam", "x.pvar", "x.bgen"):
            path = tmp_path / name
            path.write_bytes(b"\x00")

            assert "genotype-data" in rules(check_paths([path], repo_root=tmp_path)), name

    def test_a_csv_of_genotype_free_results_is_not_flagged(self, tmp_path):
        path = tmp_path / "phate_2d.csv"
        path.write_text("sample_id,dim_1,dim_2\nHG00096,0.1,0.2\n")

        assert check_paths([path], repo_root=tmp_path) == []


class TestSampleIdentifiers:
    def test_a_long_column_of_biobank_ids_is_flagged(self, tmp_path):
        # UK Biobank application IDs are seven digits. A label file is exactly
        # this shape, and one of them being stale is what started all of this.
        path = tmp_path / "project_labels.csv"
        rows = "\n".join(f"{1000000 + i},British" for i in range(80))
        path.write_text("sample_id,ancestry\n" + rows + "\n")

        assert "sample-identifiers" in rules(check_paths([path], repo_root=tmp_path))

    def test_a_short_file_is_not_flagged(self, tmp_path):
        # Tests and fixtures legitimately contain a handful of made-up ids.
        path = tmp_path / "small.csv"
        path.write_text("sample_id,ancestry\n1000001,British\n1000002,Irish\n")

        assert "sample-identifiers" not in rules(check_paths([path], repo_root=tmp_path))

    def test_public_cohort_identifiers_are_not_flagged(self, tmp_path):
        # HGDP+1KGP ids are public and appear all over the examples.
        path = tmp_path / "hgdp_labels.csv"
        rows = "\n".join(f"HG{i:05d},Africa" for i in range(80))
        path.write_text("sample_id,region\n" + rows + "\n")

        assert check_paths([path], repo_root=tmp_path) == []


class TestLocalState:
    def test_agent_and_editor_state_is_flagged(self, tmp_path):
        path = tmp_path / ".claude" / "settings.local.json"
        path.parent.mkdir()
        path.write_text("{}")

        assert "local-state" in rules(check_paths([path], repo_root=tmp_path))


class TestPrivateNames:
    def test_a_filename_saying_private_is_flagged(self, tmp_path):
        path = tmp_path / "mappings_private.json"
        path.write_text("{}")

        assert "private-name" in rules(check_paths([path], repo_root=tmp_path))


class TestClusterPaths:
    def test_an_absolute_cluster_path_in_content_is_flagged(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("Run it from /lustre06/project/6065672/someuser/ActiveProjects/x\n")

        assert "cluster-path" in rules(check_paths([path], repo_root=tmp_path))

    def test_a_relative_path_is_not_flagged(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("Run it from examples/hgdp_1kgp/\n")

        assert check_paths([path], repo_root=tmp_path) == []


class TestReporting:
    def test_a_finding_names_a_repository_relative_path(self, tmp_path):
        path = tmp_path / "sub" / "mappings_private.json"
        path.parent.mkdir()
        path.write_text("{}")

        (finding,) = check_paths([path], repo_root=tmp_path)

        assert finding.path == "sub/mappings_private.json"
        assert isinstance(finding, Finding)

    def test_a_binary_file_does_not_crash_the_content_rules(self, tmp_path):
        path = tmp_path / "blob.dat"
        path.write_bytes(bytes(range(256)))

        check_paths([path], repo_root=tmp_path)

    def test_a_missing_path_is_ignored(self, tmp_path):
        # Pre-commit passes staged paths; a deletion is staged too.
        assert check_paths([tmp_path / "gone.bed"], repo_root=tmp_path) == []

    def test_it_can_be_told_to_allow_a_path(self, tmp_path):
        """An escape hatch that is a file in the repo, not a flag someone types.

        Overriding has to leave a reviewable trace, or the check gets bypassed
        the first time it is inconvenient and stays bypassed.
        """
        path = tmp_path / "examples" / "colormaps" / "mappings_private.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        (tmp_path / ".sensitive-allow").write_text(
            "# reviewed 2026-09-12: colour definitions only, no identifiers\n"
            "examples/colormaps/mappings_private.json\n"
        )

        assert check_paths([path], repo_root=tmp_path) == []


class TestAllowlistPrecision:
    """An entry can allow one rule on a path rather than the whole path.

    Path-level exemptions are blunt: allowing a file outright means a later edit
    that adds something genuinely sensitive to it goes unnoticed. The page
    documenting this check needs exactly two rules waived and no more.
    """

    def _doc(self, tmp_path):
        path = tmp_path / "docs" / "guide.md"
        path.parent.mkdir()
        path.write_text(
            "# Private script to create UKBB sample lists\n"
            "paths under /lustre06/project are cluster-specific\n"
        )
        return path

    def test_a_rule_scoped_entry_waives_only_that_rule(self, tmp_path):
        path = self._doc(tmp_path)
        (tmp_path / ".sensitive-allow").write_text("docs/guide.md:private-marker\n")

        assert rules(check_paths([path], repo_root=tmp_path)) == {"cluster-path"}

    def test_several_scoped_entries_can_waive_several_rules(self, tmp_path):
        path = self._doc(tmp_path)
        (tmp_path / ".sensitive-allow").write_text(
            "docs/guide.md:private-marker\ndocs/guide.md:cluster-path\n"
        )

        assert check_paths([path], repo_root=tmp_path) == []

    def test_a_bare_path_entry_still_waives_everything(self, tmp_path):
        path = self._doc(tmp_path)
        (tmp_path / ".sensitive-allow").write_text("docs/guide.md\n")

        assert check_paths([path], repo_root=tmp_path) == []


class TestDataContainers:
    """Weights, caches and datasets this pipeline writes are not committed."""

    def test_model_weights_are_flagged(self, tmp_path):
        path = tmp_path / "best_model.pt"
        path.write_bytes(b"\x80")

        assert "data-container" in rules(check_paths([path], repo_root=tmp_path))

    def test_hdf5_datasets_and_genotype_caches_are_flagged(self, tmp_path):
        for name in ("dataset.hdf5", "dataset.h5", "x_genotypes.npy", "embedding.npz"):
            (tmp_path / name).write_bytes(b"\x00")

        flagged = check_paths(sorted(tmp_path.iterdir()), repo_root=tmp_path)

        assert {f.path for f in flagged if f.rule == "data-container"} == {
            "dataset.hdf5",
            "dataset.h5",
            "x_genotypes.npy",
            "embedding.npz",
        }

    def test_a_scoped_allowlist_entry_waives_only_its_rule(self, tmp_path):
        """How the committed test fixtures are allowed: per file, per rule."""
        path = tmp_path / "fixture_upstream.hdf5"
        path.write_bytes(b"\x00")
        (tmp_path / ".sensitive-allow").write_text("fixture_upstream.hdf5:data-container\n")

        assert check_paths([path], repo_root=tmp_path) == []
