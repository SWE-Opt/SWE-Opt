from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


def capture_patch_diff(repo_dir: Path, updated_files: Sequence[str]) -> str:
    """Diff patch files against HEAD using a disposable index.

    Include additions (even ignored or empty files) and deletions, excluding
    unrelated build outputs. Never modify the repository's actual index.
    Git failures propagate instead of being mistaken for an empty patch.
    """
    with tempfile.TemporaryDirectory(prefix="swe-opt-diff-") as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))

        def git(*args: str) -> str:
            result = subprocess.run(
                ["git", *args], cwd=repo_dir, env=env,
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Cannot capture patch diff: {result.stderr.strip()}"
                )
            return result.stdout

        git("read-tree", "HEAD")
        if updated_files:
            git("add", "--all", "--force", "--", *updated_files)
        return git(
            "diff", "--cached", "HEAD", "--patch", "--binary",
            "--no-ext-diff", "--no-textconv", "--",
        )