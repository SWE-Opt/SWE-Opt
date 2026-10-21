"""
patch_applier.py
----------------
Apply validated search/replace blocks to files on disk.

on_block_fail strategy
----------------------
  abort     — stop at first failure, write nothing for that file
  skip      — skip failed block, apply the rest, write what succeeded
  skip_file — if any block in a file fails, skip the entire file

ApplyStatus
-----------
  all blocks ok   → fully_applied    (ok=True)
  some blocks ok  → partially_applied (ok=False)
  nothing applied → failed           (ok=False)

Match strategy order (strictest → loosest)
------------------------------------------
  EXACT        — literal string match, unique
  DEDENTED     — textwrap.dedent applied, unique
  NORMALIZED   — strip trailing whitespace + normalize line endings, unique
  WS_COLLAPSED — collapse all whitespace runs to one space, line-based fuzzy,
                 ambiguity checked via collapsed string find()
  FUZZY        — sliding line window with variable size, margin-guarded
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from swe_opt.harness.patch.patch_sanitizer import (
    ApplyStatus,
    BlockResult,
    MatchMethod,
    PatchResult,
    SearchReplaceSanitizer,
    ValidBlock,
)

class OnBlockFail(str, Enum):
    ABORT     = "abort"
    SKIP      = "skip"
    SKIP_FILE = "skip_file"


_DEFAULT_FUZZY_THRESHOLD = 0.80
_WS_COLLAPSED_THRESHOLD  = 0.92
_FUZZY_MARGIN            = 0.05
_FUZZY_LINE_SLACK        = 6

Span = Tuple[int, int]

# Helpers
def _normalize(text: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    )

def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _span_for_lines(src_lines_ke: List[str], start: int, end: int) -> Span:
    s = sum(len(l) for l in src_lines_ke[:start])
    e = sum(len(l) for l in src_lines_ke[:end])
    return s, e

#  Match strategies

def _try_exact(src: str, search: str) -> Optional[Span]:
    idx = src.find(search)
    if idx == -1:
        return None
    if src.find(search, idx + 1) != -1:
        return None  # ambiguous
    return idx, idx + len(search)

def _try_dedented(src: str, search: str) -> Optional[Span]:
    dedented = textwrap.dedent(search)
    if dedented == search:
        return None
    idx = src.find(dedented)
    if idx == -1:
        return None
    if src.find(dedented, idx + 1) != -1:
        return None  # ambiguous
    return idx, idx + len(dedented)

def _try_normalized(src: str, search: str) -> Optional[Span]:
    norm_src    = _normalize(src)
    norm_search = _normalize(search)
    idx = norm_src.find(norm_search)
    if idx == -1:
        return None
    if norm_src.find(norm_search, idx + 1) != -1:
        return None  # ambiguous
    start_line    = norm_src[:idx].count("\n")
    matched_lines = norm_search.count("\n") + 1
    return _span_for_lines(src.splitlines(keepends=True), start_line, start_line + matched_lines)

def _try_ws_collapsed(src: str, search: str) -> Optional[Span]:
    """
    Collapse all whitespace runs to a single space, then do a
    line-based sliding window match.

    Handles:
      - multiline <-> single-line drift (Black wrapping)
      - spacing differences around = and ()
      - quote style differences (ratio-based, not exact)

    Ambiguity is checked once on fully-collapsed strings.
    No margin check — the find() ambiguity guard is sufficient.
    """
    collapsed_src    = _collapse(src)
    collapsed_search = _collapse(search)

    if not collapsed_search:
        return None

    # Ambiguity check on fully collapsed strings
    first = collapsed_src.find(collapsed_search)
    if first != -1 and collapsed_src.find(collapsed_search, first + 1) != -1:
        return None

    src_lines_ke = src.splitlines(keepends=True)
    src_lines    = src.splitlines()
    search_cl    = [_collapse(l) for l in search.splitlines() if l.strip()]

    if not search_cl:
        return None

    n          = len(search_cl)
    best_ratio = 0.0
    best_start = None
    best_end   = None

    for i in range(len(src_lines)):
        candidate = []
        j = i
        while j < len(src_lines) and len(candidate) < n:
            c = _collapse(src_lines[j])
            if c:
                candidate.append(c)
            j += 1

        if len(candidate) < n:
            break

        ratio = difflib.SequenceMatcher(None, search_cl, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end   = j

    if best_ratio < _WS_COLLAPSED_THRESHOLD or best_start is None:
        return None

    return _span_for_lines(src_lines_ke, best_start, best_end)


def _try_fuzzy(src: str, search: str, threshold: float) -> Optional[Span]:
    """
    Sliding line-window fuzzy match.
    Tries window sizes n +/- _FUZZY_LINE_SLACK to handle line-count drift.
    Requires best-vs-second margin to avoid wrong-location matches.
    """
    search_lines = search.splitlines()
    src_lines    = src.splitlines()
    src_lines_ke = src.splitlines(keepends=True)
    n            = len(search_lines)

    if n == 0 or n > len(src_lines):
        return None

    best_ratio  = 0.0
    best_start  = None
    best_n      = n
    second_best = 0.0

    for window in range(
        max(1, n - _FUZZY_LINE_SLACK),
        min(len(src_lines), n + _FUZZY_LINE_SLACK) + 1,
    ):
        for i in range(len(src_lines) - window + 1):
            ratio = difflib.SequenceMatcher(
                None, search_lines, src_lines[i : i + window]
            ).ratio()
            if ratio > best_ratio:
                second_best = best_ratio
                best_ratio  = ratio
                best_start  = i
                best_n      = window
            elif ratio > second_best:
                second_best = ratio

    if best_ratio < threshold or best_start is None:
        return None
    if best_ratio - second_best < _FUZZY_MARGIN:
        return None

    return _span_for_lines(src_lines_ke, best_start, best_start + best_n)


def _find_match(
    src: str,
    search: str,
    *,
    fuzzy: bool = True,
    fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
) -> Tuple[Optional[Span], MatchMethod]:

    span = _try_exact(src, search)
    if span:
        return span, MatchMethod.EXACT

    span = _try_dedented(src, search)
    if span:
        return span, MatchMethod.DEDENTED

    span = _try_normalized(src, search)
    if span:
        return span, MatchMethod.NORMALIZED

    span = _try_ws_collapsed(src, search)
    if span:
        return span, MatchMethod.WS_COLLAPSED

    if fuzzy:
        span = _try_fuzzy(src, search, fuzzy_threshold)
        if span:
            return span, MatchMethod.FUZZY

    return None, MatchMethod.NOT_FOUND


# Validation + formatting

def _validate_syntax(code: str) -> Optional[str]:
    try:
        import ast
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

def _format_with_black(code: str) -> Tuple[str, Optional[str]]:
    try:
        import black
        return black.format_str(code, mode=black.Mode()), None
    except ImportError:
        return code, "black not installed — skipping formatting"
    except Exception as e:
        return code, f"black: {e}"


# Indentation

def _reindent_replace(replace: str, base_indent_str: str) -> str:
    """
    Shift all lines in `replace` so that its base indentation aligns with
    `base_indent_str`. Internal relative indentation is preserved.
 
    Example:
        replace base = 4 spaces, base_indent_str = 8 spaces → delta = +4
        a line at 4  → 8
        a line at 8  → 12
        a line at 12 → 16
    """
    lines = replace.splitlines(True)
 
    # Find the base indent of the replace block itself
    replace_base_len = 0
    for line in lines:
        if line.strip():
            replace_base_len = len(line) - len(line.lstrip())
            break
 
    base_len = len(base_indent_str)
    delta    = base_len - replace_base_len  # can be negative, zero, or positive
 
    # Nothing to do
    if delta == 0:
        return replace
 
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)  # preserve blank lines as-is
            continue
        current_indent = len(line) - len(line.lstrip())
        new_indent     = max(0, current_indent + delta)
        out.append(" " * new_indent + line.lstrip())
 
    return "".join(out)

# Single block
def _apply_one_block(
    src: str,
    block: ValidBlock,
    *,
    fuzzy: bool,
    fuzzy_threshold: float,
) -> Tuple[Optional[str], MatchMethod, Optional[str]]:

    span, method = _find_match(
        src, block.search, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold
    )

    if span is None:
        return None, method, f"search block have an error in {block.file}"

    start, end = span
    matched_text = src[start:end]

    base_indent_str = ""
    for line in matched_text.splitlines():
        if line.strip():
            base_indent_str = line[: len(line) - len(line.lstrip())]
            break

    replace_base_len = 0
    for line in block.replace.splitlines():
        if line.strip():
            replace_base_len = len(line) - len(line.lstrip())
            break
    
    replace = block.replace
    if replace_base_len != len(base_indent_str):
        replace = _reindent_replace(replace, base_indent_str)
            
    if matched_text.endswith("\n") and not replace.endswith("\n"):
        replace += "\n"
    elif not matched_text.endswith("\n") and replace.endswith("\n"):
        replace = replace.rstrip("\n")

    new_src = src[:start] + replace + src[end:]

    err = _validate_syntax(new_src)
    if err:
        return None, method, f"syntax error after apply in {block.file}: {err}"

    return new_src, method, None

# File-level apply

def _apply_file_blocks(
    target: Path,
    blocks: List[ValidBlock],
    *,
    fuzzy: bool,
    fuzzy_threshold: float,
    dry_run: bool,
    run_black: bool,
    on_block_fail: OnBlockFail,
) -> Tuple[List[BlockResult], List[str], bool]:

    block_results: List[BlockResult] = []
    warnings:      List[str]         = []
    original = target.read_text(encoding="utf-8")
    current  = original

    for i, block in enumerate(blocks):
        new_src, method, error = _apply_one_block(
            current, block, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold
        )
        ok = new_src is not None
        block_results.append(BlockResult(file=block.file, match_method=method, ok=ok, error=error))

        if not ok:
            warnings.append(error or f"block failed in {block.file}")

            if on_block_fail in (OnBlockFail.ABORT, OnBlockFail.SKIP_FILE):
                mode = "abort" if on_block_fail == OnBlockFail.ABORT else "skip_file"
                for remaining in blocks[i + 1:]:
                    block_results.append(BlockResult(
                        file=remaining.file,
                        match_method=MatchMethod.NOT_FOUND,
                        ok=False,
                        error=f"skipped — earlier block failed ({mode} mode)",
                    ))
                return block_results, warnings, False

        else:
            current = new_src

    if current == original:
        return block_results, warnings, False

    if run_black and target.suffix == ".py":
        formatted, fmt_err = _format_with_black(current)
        if fmt_err:
            warnings.append(fmt_err)
        else:
            current = formatted

    if not dry_run:
        target.with_suffix(target.suffix + ".bak").write_text(original, encoding="utf-8")
        target.write_text(current, encoding="utf-8")

    return block_results, warnings, True

class SearchReplaceApplier:
    def __init__(
        self,
        *,
        repo_dir:        Path,
        dry_run:         bool,
        fuzzy:           bool        = True,
        fuzzy_threshold: float       = _DEFAULT_FUZZY_THRESHOLD,
        run_black:       bool        = True,
        on_block_fail:   OnBlockFail = OnBlockFail.ABORT,
    ):
        self.repo_dir        = repo_dir.resolve()
        self.dry_run         = dry_run
        self.fuzzy           = fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.run_black       = run_black
        self.on_block_fail   = on_block_fail

    def _resolve(self, file_rel: str) -> Tuple[Optional[Path], Optional[str]]:
        target = (self.repo_dir / file_rel).resolve()
        try:
            target.relative_to(self.repo_dir)
        except ValueError:
            return None, f"path traversal detected: {file_rel}"
        if not target.exists():
            return None, f"file not found: {file_rel}"
        return target, None

    def apply(self, blocks: List[ValidBlock], dropped_count: int = 0) -> PatchResult:
        if not blocks:
            status = ApplyStatus.FAILED if dropped_count == 0 else ApplyStatus.PARTIALLY_APPLIED
            return PatchResult(
                ok=False, apply_status=status, error="no blocks to apply",
                updated_files=[], applied_any=False, block_results=[],
                match_methods={}, dry_run=self.dry_run,
            )

        by_file: Dict[str, List[ValidBlock]] = defaultdict(list)
        for block in blocks:
            by_file[block.file].append(block)

        all_results:   List[BlockResult] = []
        all_warnings:  List[str]         = []
        updated_files: List[str]         = []

        for file_rel, file_blocks in by_file.items():
            target, err = self._resolve(file_rel)
            if target is None:
                for block in file_blocks:
                    all_results.append(BlockResult(
                        file=file_rel, match_method=MatchMethod.NOT_FOUND,
                        ok=False, error=err,
                    ))
                all_warnings.append(err or f"could not resolve {file_rel}")
                continue

            results, warnings, written = _apply_file_blocks(
                target, file_blocks,
                fuzzy=self.fuzzy,
                fuzzy_threshold=self.fuzzy_threshold,
                dry_run=self.dry_run,
                run_black=self.run_black,
                on_block_fail=self.on_block_fail,
            )
            all_results.extend(results)
            all_warnings.extend(warnings)
            if written:
                updated_files.append(file_rel)

        method_counts: Dict[str, int] = defaultdict(int)
        for r in all_results:
            method_counts[r.match_method.value] += 1

        blocks_ok     = sum(1 for r in all_results if r.ok)
        blocks_failed = sum(1 for r in all_results if not r.ok) + dropped_count

        if blocks_failed == 0:
            status, ok, error = ApplyStatus.FULLY_APPLIED, True, None
        elif blocks_ok == 0:
            status, ok = ApplyStatus.FAILED, False
            error = all_warnings[0] if all_warnings else "apply_failed"
        else:
            status, ok = ApplyStatus.PARTIALLY_APPLIED, False
            error = all_warnings[0] if all_warnings else "partially_applied"

        return PatchResult(
            ok=ok, apply_status=status, error=error,
            updated_files=updated_files, applied_any=len(updated_files) > 0,
            block_results=all_results, match_methods=dict(method_counts),
            dry_run=self.dry_run, warnings=all_warnings,
            warnings_count=len(all_warnings),
        )

class UnifiedDiffApplier:
    """Validate a Git diff, try clean application, then optionally accept hunks.

    Reject fallback runs in a temporary snapshot. It never modifies the real
    index or leaves reject files in the original repository.
    """

    def __init__(self, repo_dir: Path, diff: str, *, dry_run: bool, tolerant: bool):
        self.repo_dir = Path(repo_dir).resolve()
        self.diff = diff
        self.dry_run = dry_run
        self.tolerant = tolerant
        self.recount = False
        self.result = PatchResult(
            ok=False, apply_status=ApplyStatus.FAILED, error=None,
            updated_files=[], applied_any=False, block_results=[], match_methods={},
            kind="unified_diff", dry_run=dry_run, applied_with="git_apply",
        )

    def _git_apply(self, *options: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "apply", *(["--recount"] if self.recount else []), *options, "-"],
            cwd=cwd or self.repo_dir,
            input=self.diff, text=True, capture_output=True,
            timeout=60, check=False,
        )

    def _fail(self, error: str) -> PatchResult:
        self.result.error = error
        return self.result

    def _success(self, files: List[str], rejects: Optional[List[str]] = None) -> PatchResult:
        rejects = rejects or []
        self.result.updated_files = files
        self.result.applied_any = not self.dry_run
        self.result.warnings.extend(f"Rejected hunks in {name}" for name in rejects)
        self.result.warnings_count = len(self.result.warnings)
        self.result.ok = not rejects
        self.result.apply_status = (
            ApplyStatus.PARTIALLY_APPLIED if rejects else ApplyStatus.FULLY_APPLIED
        )
        if rejects:
            self.result.error = f"Patch partially applied; {len(rejects)} file(s) have rejected hunks"
        return self.result

    def _validate_files(self) -> List[str]:
        stats = self._git_apply("--numstat", "-z")
        if stats.returncode and "corrupt patch" in stats.stderr.lower():
            # Legacy writers stripped trailing context without updating counts.
            # Let Git recount the existing body; never invent missing code.
            self.recount = True
            self.result.tried.append("git_apply_recount")
            retry = self._git_apply("--numstat", "-z")
            if retry.returncode == 0:
                stats = retry
                self.result.warnings.append(
                    "Used git apply --recount to recover inconsistent hunk counts; "
                    "missing content is not restored"
                )
                self.result.warnings_count = len(self.result.warnings)
            else:
                self.recount = False
        if stats.returncode:
            raise ValueError(stats.stderr.strip() or "Cannot read diff file list")
        files = list(dict.fromkeys(
            entry.split("\t", 2)[2] for entry in stats.stdout.split("\0") if entry
        ))
        if not files:
            raise ValueError("diff contains no file changes")
        for filename in files:
            path = Path(filename)
            target = self.repo_dir / path
            if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
                raise ValueError(f"Unsafe patch path: {filename}")
            if target.is_symlink() or not target.resolve().is_relative_to(self.repo_dir):
                raise ValueError(f"Unsafe symlink patch path: {filename}")
        return files

    @staticmethod
    def _file_state(path: Path) -> Optional[Tuple[bytes, int]]:
        return (path.read_bytes(), path.stat().st_mode) if path.is_file() else None

    def _copy_accepted_files(self, preview: Path, files: List[str]) -> None:
        for filename in files:
            source, target = preview / filename, self.repo_dir / filename
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif target.exists():
                target.unlink()

    def _apply_three_way(self, files: List[str]) -> Optional[PatchResult]:
        """Accept only a conflict-free three-way merge in an isolated snapshot.

        Git needs the old blob IDs from the diff and those blobs locally.
        Linked worktrees are skipped: their .git file points outside the copy.
        """
        self.result.tried.append("git_apply_3way")
        git_dir = self.repo_dir / ".git"
        if not git_dir.is_dir() or git_dir.is_symlink():
            return None
        with tempfile.TemporaryDirectory(prefix="swe-opt-3way-") as directory:
            preview = Path(directory) / "repo"
            shutil.copytree(self.repo_dir, preview, symlinks=True)
            # Stage the current snapshot so Git's index/worktree consistency
            # requirement is satisfied without touching the original index.
            staged = subprocess.run(
                ["git", "add", "--", *files], cwd=preview,
                text=True, capture_output=True, timeout=60, check=False,
            )
            if staged.returncode:
                return None
            merged = self._git_apply("--3way", cwd=preview)
            self.result.stderr_tail = merged.stderr[-4000:]
            if merged.returncode:
                # Includes missing base blobs and conflicts. Discard everything.
                return None
            for filename in files:
                target = preview / filename
                if target.is_symlink() or not target.resolve().is_relative_to(preview):
                    return None
            if not self.dry_run:
                self._copy_accepted_files(preview, files)
            self.result.applied_with = "git_apply_3way"
            return self._success(files)

    def _apply_reject_fallback(self, files: List[str]) -> PatchResult:
        self.result.tried.append("git_apply_reject")
        self.result.applied_with = "git_apply_reject"
        with tempfile.TemporaryDirectory(prefix="swe-opt-reject-") as directory:
            preview = Path(directory) / "repo"
            shutil.copytree(self.repo_dir, preview, symlinks=True)
            before = {name: self._file_state(preview / name) for name in files}
            for filename in files:
                reject = preview / (filename + ".rej")
                if reject.exists() or reject.is_symlink():
                    return self._fail(f"Existing reject file would be overwritten: {filename}.rej")

            applied = self._git_apply("--reject", cwd=preview)
            self.result.stderr_tail = applied.stderr[-4000:]
            changed: List[str] = []
            rejects: List[str] = []
            for filename in files:
                target = preview / filename
                if target.is_symlink():
                    return self._fail(f"Symlink creation is not supported: {filename}")
                if before[filename] != self._file_state(target):
                    changed.append(filename)
                if (preview / (filename + ".rej")).is_file():
                    rejects.append(filename + ".rej")

            if not changed:
                return self._fail(applied.stderr.strip() or "No hunks applied")
            if applied.returncode and not rejects:
                return self._fail(applied.stderr.strip() or "Reject fallback failed")
            if not self.dry_run:
                self._copy_accepted_files(preview, changed)
            return self._success(changed, rejects)

    def apply(self) -> PatchResult:
        """Validate -> clean application -> isolated three-way -> reject fallback."""
        if not self.diff.strip():
            return self._fail("empty diff")
        try:
            files = self._validate_files()
            self.result.tried.append("git_apply")
            check = self._git_apply("--check")
            if check.returncode:
                if self.tolerant:
                    merged = self._apply_three_way(files)
                    if merged is not None:
                        return merged
                    return self._apply_reject_fallback(files)
                return self._fail(check.stderr.strip() or "git apply --check failed")
            if not self.dry_run:
                applied = self._git_apply()
                if applied.returncode:
                    return self._fail(applied.stderr.strip() or "git apply failed")
            return self._success(files)
        except (OSError, subprocess.TimeoutExpired, IndexError, ValueError, shutil.Error) as exc:
            return self._fail(str(exc))


def _apply_search_replace(
    *, repo_dir: Path, llm_output: str, dry_run: bool,
    fuzzy: bool, fuzzy_threshold: float, run_black: bool, on_block_fail: OnBlockFail,
) -> PatchResult:
    """Keep the original sanitize -> apply -> attach warnings pipeline."""
    san = SearchReplaceSanitizer.sanitize(llm_output)
    if not san["is_valid"]:
        dropped = san.get("dropped_blocks", [])
        drop_reasons = [d["reason"] for d in dropped]
        return PatchResult(
            ok=False, apply_status=ApplyStatus.FAILED,
            error=san.get("error") or "sanitizer_rejected",
            updated_files=[], applied_any=False, block_results=[], match_methods={},
            dry_run=dry_run, stage="sanitize",
            sanitizer_reason=san.get("reason"), sanitizer_dropped_blocks=dropped,
            warnings=drop_reasons, warnings_count=len(drop_reasons),
        )
    dropped = san.get("dropped_blocks", [])
    result = SearchReplaceApplier(
        repo_dir=repo_dir, dry_run=dry_run, fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold, run_black=run_black, on_block_fail=on_block_fail,
    ).apply(san["blocks"], dropped_count=len(dropped))
    result.sanitizer_reason = san.get("reason")
    result.sanitizer_dropped_blocks = dropped
    drop_reasons = [d["reason"] for d in dropped]
    if drop_reasons:
        result.warnings = drop_reasons + result.warnings
        result.warnings_count = len(result.warnings)
    return result


def apply_patch(
    *, repo_dir: Path, llm_output: str, dry_run: bool = False,
    fuzzy: bool = True, fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
    run_black: bool = True, on_block_fail: OnBlockFail = OnBlockFail.SKIP,
    tolerant_diff: bool = True,
) -> PatchResult:
    """Public format dispatcher. Both paths return the same PatchResult.

    Git diffs use clean application then optional isolated reject fallback.
    SEARCH/REPLACE uses the original sanitizer, matching and formatting rules.
    dry_run leaves the working tree unchanged on both paths.
    """
    text = (llm_output or "").lstrip()
    if text.startswith(("diff --git ", "--- ")):
        return UnifiedDiffApplier(
            repo_dir, text, dry_run=dry_run, tolerant=tolerant_diff,
        ).apply()
    return _apply_search_replace(
        repo_dir=repo_dir, llm_output=llm_output, dry_run=dry_run,
        fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold,
        run_black=run_black, on_block_fail=on_block_fail,
    )