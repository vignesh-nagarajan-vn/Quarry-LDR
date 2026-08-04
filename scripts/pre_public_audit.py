"""Go/no-go audit before this repo is made public.

Scans the full git history for secret patterns, checks every tracked file for
absolute personal paths and identifiers, verifies license presence, verifies
fixtures are synthetic (provenance marker emitted by make_fixtures.py),
and enforces the no-em-dash rule in README.md, CLAUDE.md, and COMMIT.md.
Prints a verdict and exits nonzero on no-go.

Structure: every check below is a pure function of a repo root (or, for the
git-history check, of pre-fetched `git log -p` text) so each one is directly
unit-testable against a constructed tmp_path tree with no subprocess and no
real git repository required. ``main()`` is the only place that shells out to
git and the only place that prints; it composes the checks and reports a
final PASS/FAIL table plus a GO / NO-GO verdict.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Anthropic-style API keys, anywhere in a file or a git diff line.
SECRET_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")
PRIVATE_KEY_MARKERS = (  # pragma: allowlist secret (detection patterns, not keys)
    "BEGIN RSA PRIVATE KEY",  # pragma: allowlist secret
    "BEGIN OPENSSH PRIVATE KEY",  # pragma: allowlist secret
)

# A matched line is not a leak when it carries one of these markers, or when
# the match itself is the literal string quarry_ldr.logging.redact() produces
# for a real key ("sk-ant-REDACTED", used both in the redaction table and in
# tests that assert redaction happened).
ALLOWLIST_MARKERS = ("not-a-real-key", "pragma: allowlist secret")
REDACTED_LITERAL = "sk-ant-REDACTED"
# Exact synthetic fixtures committed before their allowlist pragma existed;
# history blobs cannot be re-annotated, so these transparent fakes are
# allowlisted by exact literal (never by pattern).
HISTORY_FIXTURE_LITERALS = ("sk-ant-api03-" + "abcdef1234567890",)  # pragma: allowlist secret

WINDOWS_USER_PATH = re.compile(r"[A-Za-z]:\\Users\\(?!nrvig-free-placeholder)")
UNIX_HOME_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+/")

EM_DASH = "\u2014"
EM_DASH_FILES = (
    "README.md",
    "CLAUDE.md",
    "COMMIT.md",
    "docs/Architecture.md",
    "docs/Troubleshooting.md",
)

# Directories that are never git-tracked source: caches, venvs, model/data
# blobs, and this dev machine's local lancedb scratch dirs. Approximating
# "tracked files" this way (rather than shelling out to `git ls-files`) keeps
# every check below a pure function that unit tests can point at a plain
# tmp_path tree, with no .git directory required.
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    "htmlcov",
    "models",
    "data",
    "logs",
    ".idea",
    ".vscode",
}


def _should_skip_dir(name: str) -> bool:
    return (
        name in _EXCLUDED_DIR_NAMES
        or name.endswith(".egg-info")
        or name.endswith(".lance")
        or name.startswith("scratch_")
    )


def _iter_scannable_files(repo_root: Path) -> Iterator[Path]:
    """Walk repo_root yielding candidate text files, skipping build/cache/
    model/data directories. Binary files are filtered out by callers via a
    failed utf-8 decode, not here."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for filename in filenames:
            yield Path(dirpath) / filename


def _read_text(path: Path) -> str | None:
    """Best-effort utf-8 read; None for anything that is not a text file."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


# --------------------------------------------------------------------- (a)


def check_no_secrets_in_tree(repo_root: Path) -> CheckResult:
    """Tracked text files must carry no Anthropic-shaped key. `.env` is
    exempt alongside `.env.example`: on a configured machine it legitimately
    holds the real key, and check (g) fails the audit on its own unless
    `.env` is gitignored, so the overall verdict stays safe. Matched secret
    content is never echoed; only its location is reported."""
    violations: list[str] = []
    for path in _iter_scannable_files(repo_root):
        if path.name in (".env", ".env.example"):
            continue
        text = _read_text(path)
        if text is None:
            continue
        rel = path.relative_to(repo_root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in ALLOWLIST_MARKERS):
                continue
            for match in SECRET_PATTERN.finditer(line):
                if match.group(0) == REDACTED_LITERAL:
                    continue
                violations.append(f"{rel}:{lineno}: anthropic-key-shaped string (redacted)")
    passed = not violations
    detail = "no secret-shaped strings in tracked files" if passed else "; ".join(violations)
    return CheckResult("no secrets in working tree", passed, detail)


# --------------------------------------------------------------------- (b)


def check_no_secrets_in_history(history_text: str) -> CheckResult:
    """Scan `git log -p` output for the same key pattern plus private-key
    headers. Pure function of the diff text: main() is the only caller that
    actually runs git, so this is unit-testable with a crafted string and no
    subprocess."""
    violations: list[str] = []
    current_file = ""
    for lineno, line in enumerate(history_text.splitlines(), start=1):
        if line.startswith(("+++ ", "--- ")):
            path_part = line[4:].split("\t", 1)[0].strip()
            if path_part.startswith(("a/", "b/")):
                path_part = path_part[2:]
            current_file = path_part
            continue
        if current_file == ".env.example":
            continue
        if any(marker in line for marker in ALLOWLIST_MARKERS):
            continue
        match = SECRET_PATTERN.search(line)
        if (
            match is not None
            and match.group(0) != REDACTED_LITERAL
            and match.group(0) not in HISTORY_FIXTURE_LITERALS
        ):
            violations.append(
                f"line {lineno} ({current_file}): anthropic-key-shaped string (redacted)"
            )
            continue
        for marker in PRIVATE_KEY_MARKERS:
            if marker in line:
                violations.append(f"line {lineno} ({current_file}): {marker}")
    passed = not violations
    detail = (
        "no secret-shaped strings or private key headers in history"
        if passed
        else "; ".join(violations)
    )
    return CheckResult("no secrets in git history", passed, detail)


def _run_git_log_p(repo_root: Path) -> str | None:
    """Read-only: `git log -p`. None (not a failure) when git is unavailable
    or repo_root is not a git repository at all."""
    try:
        result = subprocess.run(
            ["git", "log", "-p"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


# --------------------------------------------------------------------- (c)


def check_no_personal_paths(repo_root: Path) -> CheckResult:
    """No Windows `C:\\Users\\...` or Unix `/home/<name>/` path in any
    tracked file. Only file contents are scanned: git author names and
    emails in commit metadata are normal repository history, not a leak, and
    are deliberately never inspected here."""
    violations: list[str] = []
    for path in _iter_scannable_files(repo_root):
        if path.name == "uv.lock":
            continue
        text = _read_text(path)
        if text is None:
            continue
        rel = path.relative_to(repo_root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if WINDOWS_USER_PATH.search(line) or UNIX_HOME_PATH.search(line):
                violations.append(f"{rel}:{lineno}: {line.strip()[:160]}")
    passed = not violations
    detail = "no personal absolute paths in tracked files" if passed else "; ".join(violations)
    return CheckResult("no personal absolute paths", passed, detail)


# --------------------------------------------------------------------- (d)


def check_license(repo_root: Path) -> CheckResult:
    path = repo_root / "LICENSE"
    if not path.is_file():
        return CheckResult("LICENSE present (MIT)", False, f"{path} does not exist")
    text = _read_text(path) or ""
    passed = "MIT" in text
    detail = f"{path} present and mentions MIT" if passed else f"{path} exists but has no MIT text"
    return CheckResult("LICENSE present (MIT)", passed, detail)


# --------------------------------------------------------------------- (e)


def _find_urls(data: object) -> Iterator[str]:
    """Every string value stored under a "url" key, anywhere in the tree."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "url" and isinstance(value, str):
                yield value
            else:
                yield from _find_urls(value)
    elif isinstance(data, list):
        for item in data:
            yield from _find_urls(item)


def check_fixture_provenance(repo_root: Path) -> CheckResult:
    provenance = repo_root / "tests" / "fixtures" / "PROVENANCE.md"
    manifest = repo_root / "tests" / "fixtures" / "manifest.json"
    problems: list[str] = []

    if not provenance.is_file():
        problems.append(f"{provenance} is missing")
    elif "SYNTHETIC-FIXTURE-CORPUS" not in (_read_text(provenance) or ""):
        problems.append(f"{provenance} is missing the SYNTHETIC-FIXTURE-CORPUS marker")

    if not manifest.is_file():
        problems.append(f"{manifest} is missing")
    else:
        raw = _read_text(manifest)
        try:
            data = json.loads(raw) if raw is not None else None
        except json.JSONDecodeError as exc:
            problems.append(f"{manifest} is not valid JSON: {exc}")
            data = None
        if data is not None:
            bad_urls = [url for url in _find_urls(data) if not _hostname(url).endswith(".example")]
            if bad_urls:
                problems.append(
                    "non-.example URLs in manifest: " + ", ".join(sorted(set(bad_urls))[:5])
                )

    passed = not problems
    detail = "fixtures are synthetic with .example URLs" if passed else "; ".join(problems)
    return CheckResult("fixture provenance is synthetic", passed, detail)


def _hostname(url: str) -> str:
    return urlsplit(url).hostname or ""


# --------------------------------------------------------------------- (f)


def check_no_em_dash(repo_root: Path) -> CheckResult:
    violations: list[str] = []
    for name in EM_DASH_FILES:
        path = repo_root / name
        text = _read_text(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line:
                violations.append(f"{name}:{lineno}")
    passed = not violations
    detail = "no em dash in README/CLAUDE/COMMIT" if passed else "; ".join(violations)
    return CheckResult("no em dash (U+2014)", passed, detail)


# --------------------------------------------------------------------- (g)


def check_env_gitignored(repo_root: Path) -> CheckResult:
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return CheckResult(".env is gitignored", False, f"{gitignore} does not exist")
    text = _read_text(gitignore) or ""
    patterns = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    ignored = any(_gitignore_pattern_matches_env(p) for p in patterns)
    detail = (
        "`.env` matched by a .gitignore pattern"
        if ignored
        else "no .gitignore pattern covers `.env`; add a `.env` line"
    )
    return CheckResult(".env is gitignored", ignored, detail)


def _gitignore_pattern_matches_env(pattern: str) -> bool:
    candidate = pattern.lstrip("/")
    if candidate in (".env", ".env/"):
        return True
    if candidate.endswith("/"):
        # Directory-only pattern: cannot match a plain file named .env.
        return False
    return fnmatch.fnmatch(".env", candidate)


# ------------------------------------------------------------------- main


def run_all_checks(repo_root: Path) -> list[CheckResult]:
    """Every check, including the git-history scan (main()'s subprocess
    boundary lives here, not inside any individual check function)."""
    results = [check_no_secrets_in_tree(repo_root)]
    history_text = _run_git_log_p(repo_root)
    if history_text is None:
        results.append(
            CheckResult(
                "no secrets in git history",
                False,
                "could not run `git log -p` (not a git repository, or git is not on PATH)",
            )
        )
    else:
        results.append(check_no_secrets_in_history(history_text))
    results.extend(
        [
            check_no_personal_paths(repo_root),
            check_license(repo_root),
            check_fixture_provenance(repo_root),
            check_no_em_dash(repo_root),
            check_env_gitignored(repo_root),
        ]
    )
    return results


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    results = run_all_checks(repo_root)

    print("pre-public audit")
    print("=" * 60)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        if not result.passed:
            print(f"       {result.detail}")

    go = all(result.passed for result in results)
    print("=" * 60)
    print("GO" if go else "NO-GO")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
