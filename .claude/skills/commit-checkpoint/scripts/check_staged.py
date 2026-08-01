#!/usr/bin/env python3
"""Block sensitive paths and hardcoded secrets from being committed.

Run with changes already staged:

    python .claude/skills/commit-checkpoint/scripts/check_staged.py

Exit status is 0 when the staged set is clean and 1 when something needs
attention, so it can also be wired into a pre-commit hook later.

Content scanning is deliberately limited to Python sources and the plain-text
files under admin_files/. Those are where this project's secrets have actually
appeared, and narrowing the scope keeps false positives near zero - a guard that
cries wolf gets ignored, which defeats the point.
"""

import re
import subprocess
import sys

# Paths that must never enter a commit, matched against the repo-relative path.
FORBIDDEN_PATHS = [
    (re.compile(r"^face-files/"), "enrolled face images (biometric data)"),
    # The layout before images moved under face-files/.
    (re.compile(r"^s\d+/"), "enrolled face images (biometric data, old layout)"),
    (re.compile(r"^db\.sqlite3$"), "local database (real accounts and password hashes)"),
    (re.compile(r"^trainer\.yml$"), "LBPH model derived from face images"),
]

# Secret shapes, checked only in the files listed by _should_scan_content.
SECRET_PATTERNS = [
    (
        re.compile(r"""SECRET_KEY\s*=\s*['"][^'"]{16,}['"]"""),
        "hardcoded SECRET_KEY - read DJANGO_SECRET_KEY from the environment",
    ),
    (
        re.compile(r"""['"][A-Za-z0-9]{40,}['"]"""),
        "long literal string that looks like an API key",
    ),
]


def _git(*args):
    """Run a git command and return stdout, or None if git failed."""
    try:
        out = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _staged_paths():
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACM")
    if out is None:
        return None
    return [p.strip() for p in out.splitlines() if p.strip()]


def _should_scan_content(path):
    return path.endswith(".py") or path.startswith("admin_files/")


def _staged_content(path):
    """Read the staged (index) version, which may differ from the worktree."""
    return _git("show", ":" + path)


def main():
    paths = _staged_paths()
    if paths is None:
        print("check_staged: not a git repository, or git is unavailable.")
        return 1
    if not paths:
        print("check_staged: nothing staged - stage your changes first.")
        return 0

    problems = []

    for path in paths:
        for pattern, reason in FORBIDDEN_PATHS:
            if pattern.search(path):
                problems.append((path, reason))

        if not _should_scan_content(path):
            continue

        content = _staged_content(path)
        if content is None:
            continue

        # A number here is someone's phone number, set through the UI at runtime.
        if path == "admin_files/mobile_no.txt" and content.strip():
            problems.append((path, "contains a phone number - commit it empty"))
            continue

        for pattern, reason in SECRET_PATTERNS:
            for match in pattern.finditer(content):
                line = content[: match.start()].count("\n") + 1
                problems.append(("%s:%d" % (path, line), reason))

    if problems:
        print("check_staged: %d problem(s) found in staged changes\n" % len(problems))
        for where, reason in problems:
            print("  %-44s %s" % (where, reason))
        print(
            "\nUnstage with:  git restore --staged <path>"
            "\nIf it should never be tracked, add it to .gitignore too."
        )
        return 1

    print("check_staged: %d staged file(s), nothing sensitive found." % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
