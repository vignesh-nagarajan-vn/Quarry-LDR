"""Unit tests for scripts/pre_public_audit.py and scripts/smoke.py.

Every pre_public_audit check is a pure function of a repo root, so each one
is exercised here against small tmp_path trees built for that one check.
The git-history scanner is a pure function of `git log -p`-shaped text
instead: it is fed crafted strings, never a real subprocess, per the "no
subprocess in unit tests" rule; main()'s composition of it is covered
separately by monkeypatching the subprocess boundary.

smoke.py is never run end to end here (no ANTHROPIC_API_KEY in this
environment, by design); only its pure helpers (preflight, validate_citations)
are exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from quarry_ldr.config import QuarryConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import pre_public_audit as audit
import smoke


def _write_manifest(path: Path, urls: list[str]) -> None:
    path.write_text(json.dumps({"docs": [{"url": u} for u in urls]}), encoding="utf-8")


# --------------------------------------------------------- (a) secrets, tree


def test_no_secrets_in_tree_flags_planted_key(tmp_path: Path) -> None:
    (tmp_path / "leak.py").write_text(
        'API_KEY = "sk-ant-abcdef1234567890"\n',  # pragma: allowlist secret (planted)
        encoding="utf-8",
    )
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is False
    assert "leak.py" in result.detail


def test_no_secrets_in_tree_passes_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("nothing secret here\n", encoding="utf-8")
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is True


def test_no_secrets_in_tree_allows_env_example_placeholder(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text(
        "ANTHROPIC_API_KEY=sk-ant-your-key-here\n",  # pragma: allowlist secret (planted)
        encoding="utf-8",
    )
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is True


def test_no_secrets_in_tree_allows_pragma_marker(tmp_path: Path) -> None:
    (tmp_path / "test_x.py").write_text(
        'fake_key = "sk-ant-abcdef1234567890"  # pragma: allowlist secret\n', encoding="utf-8"
    )
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is True


def test_no_secrets_in_tree_allows_not_a_real_key_marker(tmp_path: Path) -> None:
    (tmp_path / "test_y.py").write_text(
        'fake_key = "sk-ant-test-not-a-real-key"\n',
        encoding="utf-8",  # pragma: allowlist secret (planted test fixture)
    )
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is True


def test_no_secrets_in_tree_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(
        b"\xff\xfe\x00sk-ant-abcdef1234567890",  # pragma: allowlist secret (planted)
    )
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is True


def test_no_secrets_in_tree_skips_excluded_dirs(tmp_path: Path) -> None:
    venv = tmp_path / ".venv" / "site-packages"
    venv.mkdir(parents=True)
    (venv / "leak.py").write_text(
        '"sk-ant-abcdef1234567890"\n',  # pragma: allowlist secret (planted)
        encoding="utf-8",
    )
    result = audit.check_no_secrets_in_tree(tmp_path)
    assert result.passed is True


# ----------------------------------------------------- (b) secrets, history


def test_no_secrets_in_history_flags_leaked_key() -> None:
    history = "\n".join(
        [
            "commit abc123",
            "Author: Someone <someone@example.com>",
            "",
            "    add config",
            "",
            "diff --git a/leak.txt b/leak.txt",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/leak.txt",
            "@@ -0,0 +1 @@",
            "+ANTHROPIC_API_KEY=sk-ant-abcdef1234567890",  # pragma: allowlist secret (planted test fixture)
        ]
    )
    result = audit.check_no_secrets_in_history(history)
    assert result.passed is False
    assert "leak.txt" in result.detail


def test_no_secrets_in_history_flags_private_key_header() -> None:
    history = "\n".join(
        [
            "diff --git a/id_rsa b/id_rsa",
            "--- /dev/null",
            "+++ b/id_rsa",
            "+-----BEGIN RSA PRIVATE KEY-----",  # pragma: allowlist secret (planted)
        ]
    )
    result = audit.check_no_secrets_in_history(history)
    assert result.passed is False
    assert "BEGIN RSA PRIVATE KEY" in result.detail  # pragma: allowlist secret


def test_no_secrets_in_history_allows_env_example_placeholder() -> None:
    history = "\n".join(
        [
            "diff --git a/.env.example b/.env.example",
            "--- a/.env.example",
            "+++ b/.env.example",
            "+ANTHROPIC_API_KEY=sk-ant-your-key-here",  # pragma: allowlist secret (planted test fixture)
        ]
    )
    result = audit.check_no_secrets_in_history(history)
    assert result.passed is True


def test_no_secrets_in_history_allows_pragma_marker() -> None:
    history = "\n".join(
        [
            "diff --git a/tests/unit/test_config.py b/tests/unit/test_config.py",
            "--- a/tests/unit/test_config.py",
            "+++ b/tests/unit/test_config.py",
            '+    fake_key = "sk-ant-test-not-a-real-key"  # pragma: allowlist secret',
        ]
    )
    result = audit.check_no_secrets_in_history(history)
    assert result.passed is True


def test_run_all_checks_composes_history_scan_via_subprocess_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main()'s only subprocess call is _run_git_log_p; run_all_checks()
    composes it. Verified here by monkeypatching that one boundary function,
    never by actually running git."""
    monkeypatch.setattr(audit, "_run_git_log_p", lambda repo_root: None)
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "PROVENANCE.md").write_text("SYNTHETIC-FIXTURE-CORPUS\n", encoding="utf-8")
    _write_manifest(fixtures / "manifest.json", ["https://a.example/1"])

    results = audit.run_all_checks(tmp_path)

    history_result = next(r for r in results if r.name == "no secrets in git history")
    assert history_result.passed is False
    assert "git log -p" in history_result.detail
    # every other check still ran and passed against the tree built above
    assert all(r.passed for r in results if r.name != "no secrets in git history")


# -------------------------------------------------------- (c) personal paths


def test_no_personal_paths_flags_windows_user_dir(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(
        "cache lives at C:" + "\\Users" + "\\someone\\AppData\\Local\\quarry" + "\n",
        encoding="utf-8",
    )
    result = audit.check_no_personal_paths(tmp_path)
    assert result.passed is False
    assert "notes.txt" in result.detail


def test_no_personal_paths_flags_unix_home_dir(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(
        "data lives at /home" + "/someone/quarry\n", encoding="utf-8"
    )
    result = audit.check_no_personal_paths(tmp_path)
    assert result.passed is False


def test_no_personal_paths_passes_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("cache lives at data/cache\n", encoding="utf-8")
    result = audit.check_no_personal_paths(tmp_path)
    assert result.passed is True


def test_no_personal_paths_ignores_uv_lock(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(
        "C:" + "\\Users" + "\\someone\\pip-cache" + "\n", encoding="utf-8"
    )
    result = audit.check_no_personal_paths(tmp_path)
    assert result.passed is True


# --------------------------------------------------------------- (d) LICENSE


def test_license_missing(tmp_path: Path) -> None:
    result = audit.check_license(tmp_path)
    assert result.passed is False


def test_license_present_without_mit(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text("All rights reserved.\n", encoding="utf-8")
    result = audit.check_license(tmp_path)
    assert result.passed is False


def test_license_present_with_mit(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted.\n", encoding="utf-8"
    )
    result = audit.check_license(tmp_path)
    assert result.passed is True


# ----------------------------------------------------- (e) fixture provenance


def test_fixture_provenance_missing_files(tmp_path: Path) -> None:
    result = audit.check_fixture_provenance(tmp_path)
    assert result.passed is False


def test_fixture_provenance_missing_marker(tmp_path: Path) -> None:
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "PROVENANCE.md").write_text("nothing to see here\n", encoding="utf-8")
    _write_manifest(fixtures / "manifest.json", ["https://a.example/1"])
    result = audit.check_fixture_provenance(tmp_path)
    assert result.passed is False
    assert "SYNTHETIC-FIXTURE-CORPUS" in result.detail


def test_fixture_provenance_flags_non_example_url(tmp_path: Path) -> None:
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "PROVENANCE.md").write_text("SYNTHETIC-FIXTURE-CORPUS\n", encoding="utf-8")
    _write_manifest(fixtures / "manifest.json", ["https://real-looking-site.com/article"])
    result = audit.check_fixture_provenance(tmp_path)
    assert result.passed is False
    assert "real-looking-site.com" in result.detail


def test_fixture_provenance_passes_clean_synthetic_tree(tmp_path: Path) -> None:
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "PROVENANCE.md").write_text("SYNTHETIC-FIXTURE-CORPUS\n", encoding="utf-8")
    _write_manifest(fixtures / "manifest.json", ["https://a.example/1", "https://b.example/2"])
    result = audit.check_fixture_provenance(tmp_path)
    assert result.passed is True


def test_fixture_provenance_passes_on_real_repo(repo_root: Path) -> None:
    """The real tests/fixtures corpus must satisfy its own provenance check."""
    result = audit.check_fixture_provenance(repo_root)
    assert result.passed is True, result.detail


# --------------------------------------------------------------- (f) em dash


def test_no_em_dash_flags_planted_dash(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Quarry \u2014 a research tool\n", encoding="utf-8")
    result = audit.check_no_em_dash(tmp_path)
    assert result.passed is False
    assert "README.md" in result.detail


def test_no_em_dash_passes_clean_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Quarry: a research tool\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Operating rules.\n", encoding="utf-8")
    (tmp_path / "COMMIT.md").write_text("Commit contract.\n", encoding="utf-8")
    result = audit.check_no_em_dash(tmp_path)
    assert result.passed is True


def test_no_em_dash_ignores_files_outside_the_checked_set(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("an em dash \u2014 here is fine\n", encoding="utf-8")
    result = audit.check_no_em_dash(tmp_path)
    assert result.passed is True


# ------------------------------------------------------- (g) .env gitignored


def test_env_gitignored_true(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".env\ndata/\n", encoding="utf-8")
    result = audit.check_env_gitignored(tmp_path)
    assert result.passed is True


def test_env_gitignored_false_no_matching_pattern(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("data/\n", encoding="utf-8")
    result = audit.check_env_gitignored(tmp_path)
    assert result.passed is False


def test_env_gitignored_false_missing_file(tmp_path: Path) -> None:
    result = audit.check_env_gitignored(tmp_path)
    assert result.passed is False


def test_env_gitignored_wildcard_pattern(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".env*\n", encoding="utf-8")
    result = audit.check_env_gitignored(tmp_path)
    assert result.passed is True


# -------------------------------------------------------------- smoke.py


def test_smoke_preflight_flags_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = QuarryConfig(_env_file=None)
    checks = smoke.preflight(cfg)
    api_key_check = next(c for c in checks if c.name == "ANTHROPIC_API_KEY")
    assert api_key_check.ok is False


def test_smoke_main_exits_2_when_preflight_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(smoke, "build_config", lambda: QuarryConfig(_env_file=None))
    monkeypatch.setattr(
        smoke,
        "preflight",
        lambda cfg: [smoke.PreflightCheck("ANTHROPIC_API_KEY", False, "not set")],
    )
    assert smoke.main() == 2


def test_smoke_main_proceeds_past_preflight_when_all_checks_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure control-flow check: when every preflight item is ok, main()
    proceeds past preflight (never reachable in this key-less environment
    without also mocking the whole pipeline run, which smoke.py is
    deliberately never exercised against here)."""
    monkeypatch.setattr(smoke, "build_config", lambda: QuarryConfig(_env_file=None))
    monkeypatch.setattr(
        smoke,
        "preflight",
        lambda cfg: [smoke.PreflightCheck("ANTHROPIC_API_KEY", True, "found in environment")],
    )

    def _boom(cfg: QuarryConfig) -> tuple[int, Path | None]:
        raise AssertionError("orchestrator should only run once preflight fully passes")

    # _run is the async worker main() drives via asyncio.run; monkeypatching
    # it to raise proves control flow actually left the preflight branch.
    monkeypatch.setattr(smoke, "_run", _boom)
    with pytest.raises(AssertionError):
        smoke.main()


def test_validate_citations_passes_when_all_resolve() -> None:
    report = (
        "# Topic\n\nSome claim [1] and another [2].\n\n"
        "## References\n\n[1] Source A - https://a.example\n[2] Source B - https://b.example\n"
    )
    assert smoke.validate_citations(report) == []


def test_validate_citations_flags_unresolved_marker() -> None:
    report = (
        "# Topic\n\nSome claim [1] and another [3].\n\n"
        "## References\n\n[1] Source A - https://a.example\n[2] Source B - https://b.example\n"
    )
    problems = smoke.validate_citations(report)
    assert problems
    assert "3" in problems[0]


def test_validate_citations_requires_references_section() -> None:
    report = "# Topic\n\nSome claim [1].\n"
    assert smoke.validate_citations(report) == ["report has no '## References' section"]
