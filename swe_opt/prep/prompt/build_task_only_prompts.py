"""
- The benchmark's original task description.
- The public entry point.
- The benchmark-wide optimization objectives:
  runtime, peak memory, and time-weighted mean memory.
- Instructions allowing the agent to inspect, edit, and test the repository.

The prompt does not contain oracle source files, oracle file contents, or
oracle file paths. The agent must locate the relevant implementation by
exploring the repository.

The inference runner (script 2) consumes these prompt files. This script does
not launch Docker, run OpenCode, or call a model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from swe_opt.prep.data_loader import DatasetLoader


logger = logging.getLogger("swe_opt.agentic_prompt_build")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


AGENT_ACTION = """\
WORKING ENVIRONMENT
- The current working directory is the repository root.
- The repository is checked out at the required baseline commit.
- Project dependencies and test tools are already installed.
- Inspect and edit repository source files directly.
- Use repository-relative paths.
- Run focused existing tests when useful.
- Do not install or update dependencies.
- Do not modify tests, benchmarks, fixtures, configuration, or expected outputs.
- Do not create benchmark, profiling, scratch, or explanatory files.
- Leave only source-code changes required for the optimization.
- Do not print a patch; make the changes directly in the working tree.

BUDGET
You have a maximum of 50 agent turns. Manage the available turns carefully.
Prioritize understanding the relevant execution path, implementing the change,
and validating it with focused tests. Stop when the implementation is complete.
"""

def safe_id(value: str) -> str:
    """
    Convert an identifier into a safe directory name.

    A short hash is appended to prevent two distinct identifiers that normalize
    to the same text from silently sharing an output directory.
    """
    original = str(value).strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "__", original)
    normalized = normalized.strip("._-")

    if not normalized:
        normalized = "unknown"

    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    return f"{normalized}__{digest}"


def clean_text(value: Any) -> str:
    """Return a stripped string, treating None as an empty value."""
    if value is None:
        return ""

    return str(value).strip()


def get_scenario_id(row: dict[str, Any], row_number: int) -> str:
    """
    Get a scenario identifier.

    Missing identifiers receive a deterministic row-based fallback so multiple
    missing IDs cannot overwrite one another.
    """
    scenario_id = clean_text(row.get("scenario_id"))

    if scenario_id:
        return scenario_id

    return f"unknown_instance_{row_number:06d}"


def build_prompt_prefix(row: dict[str, Any]) -> str:
    """
    Build the task specification without oracle context.

    The original benchmark task is preserved verbatim apart from trimming
    surrounding whitespace. The optimization metrics are benchmark-wide and
    therefore do not reveal scenario-specific implementation information.
    """
    task = clean_text(row.get("task"))
    entry_point = clean_text(row.get("entry_point"))

    parts: list[str] = [
        "You are an expert software performance engineer.",
        "",
        "OBJECTIVE:",
        "Optimize the requested operation according to all three benchmark "
        "metrics:",
        "- Runtime: reduce execution time.",
        "- Peak memory: reduce the maximum memory used during execution.",
        "- Time-weighted mean memory: reduce memory usage over the full "
        "execution period.",
        "",
        "Treat all three metrics as optimization objectives. Preserve externally "
        "observable behavior, and do not improve one metric through a substantial "
        "regression in another.",
        "",
        "BENCHMARK TASK:",
        task,
    ]

    if entry_point:
        parts.extend(
            [
                "",
                "ENTRY POINT:",
                entry_point,
            ]
        )

    parts.extend(
        [
            "",
            "CONSTRAINTS:",
            "- Preserve exact observable behavior and outputs.\n"
            "- Preserve public APIs, signatures, and return types.\n"
            "- Preserve existing test expectations.\n"
            "- Avoid changes unrelated to the requested optimization."
        ]
    )

    return "\n".join(parts).rstrip() + "\n"


def build_agentic_prompt(row: dict[str, Any]) -> str:
    """Build the complete task-only prompt for one benchmark scenario."""
    prefix = build_prompt_prefix(row)
    return f"{prefix}\n{AGENT_ACTION}".rstrip() + "\n"


def write_json(path: Path, value: Any) -> None:
    """Write a JSON value with stable, human-readable formatting."""
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog="Agentic task-only prompt builder",
        description=(
            "Build task-only prompts for an OpenCode agentic benchmark run. "
            "Datasets may be local JSON/JSONL files or Hugging Face dataset IDs."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="SWE-Opt/SWE-Opt",
        help=(
            "local JSON/JSONL dataset path or Hugging Face dataset ID "
            "(Hugging Face datasets use the 'test' split)"
        ),
    )

    parser.add_argument(
        "--out_root",
        type=Path,
        default="out",
        help=(
            "prompts are written under "
            "<out_root>/<run_id>/instances/<scenario_id>/"
        ),
    )

    parser.add_argument(
        "--run_id",
        default="opencode__agentic__task_only",
        help="unique identifier used to create a non-overwriting prompt run directory",
    )

    return parser.parse_args()


def validate_args(args: Namespace) -> None:
    """Validate arguments before loading a local or Hugging Face dataset."""
    if not clean_text(args.dataset_path):
        raise ValueError("--dataset_path must not be empty")

    if not clean_text(args.run_id):
        raise ValueError("--run_id must not be empty")


def main() -> None:
    args = parse_args()
    validate_args(args)

    loader = DatasetLoader(str(args.dataset_path), mode="raw")

    run_dir = args.out_root / safe_id(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    instances_root = run_dir / "instances"
    instances_root.mkdir()

    metadata = {
        "run_id": args.run_id,
        "pipeline": "agentic",
        "agent": "opencode",
        "context": "task_only_unaugmented",
        "dataset_path": str(args.dataset_path),
        "optimization_metrics": [
            "runtime",
            "peak_memory",
            "time_weighted_mean_memory",
        ],
        "oracle_file_contents_in_prompt": False,
        "oracle_file_paths_in_prompt": False,
        "task_descriptions_augmented": False,
    }

    write_json(run_dir / "PROMPT_META.json", metadata)

    count = 0
    seen_scenario_ids: set[str] = set()

    for row_number, row in enumerate(loader):
        if not isinstance(row, dict):
            raise TypeError(
                f"Dataset row {row_number} must be a dictionary, "
                f"got {type(row).__name__}"
            )

        scenario_id = get_scenario_id(row, row_number)

        if scenario_id in seen_scenario_ids:
            raise ValueError(
                f"Duplicate scenario_id at row {row_number}: "
                f"{scenario_id!r}"
            )

        seen_scenario_ids.add(scenario_id)

        prompt = build_agentic_prompt(row)
        instance_dir = instances_root / safe_id(scenario_id)

        if instance_dir.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing instance directory: "
                f"{instance_dir}"
            )

        instance_dir.mkdir(parents=True)

        (instance_dir / "prompt.md").write_text(
            prompt,
            encoding="utf-8",
        )

        # Preserve the complete original row so the inference runner can access
        # repository, Docker image, baseline SHA, and evaluation information.
        write_json(instance_dir / "row.json", row)

        logger.info("%s: wrote prompt.md", scenario_id)
        count += 1

    logger.info(
        "Done: wrote %d prompt(s). Prompt run directory: %s",
        count,
        run_dir,
    )


if __name__ == "__main__":
    main()