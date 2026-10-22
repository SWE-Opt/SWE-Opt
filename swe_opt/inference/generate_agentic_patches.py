"""Generate one OpenCode patch per scenario inside docker.baseline.

The agent never receives docker.reference. Each immutable baseline image
already contains the repository and its tested dependencies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("swe_opt.agentic_gen")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DOCKER_PLATFORM = "linux/amd64"
MAX_AGENT_STEPS = 50
ASSISTANT_PREFILL_ERROR = "does not support assistant message prefill"


def safe_id(value: str) -> str:
    return str(value).replace("/", "__").replace(":", "_").strip()


def get_scenario_id(row: Dict[str, Any]) -> str:
    value = row.get("scenario_id")
    return str(value).strip() if value and str(value).strip() else "unknown_instance"


def run_command(
    command: Sequence[str],
    *,
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}"
        )
    return result


def get_baseline_image(row: Dict[str, Any]) -> tuple[str, str, str]:
    docker = row.get("docker")
    if not isinstance(docker, dict):
        raise ValueError("row.json is missing docker")
    baseline = docker.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("row.json is missing docker.baseline")

    tag = str(baseline.get("tag") or "").strip()
    digest = str(baseline.get("digest") or "").strip()
    if not tag:
        raise ValueError("row.json is missing docker.baseline.tag")
    if digest and not digest.startswith("sha256:"):
        raise ValueError(f"Invalid docker.baseline.digest: {digest}")
    return tag, digest, f"{tag}@{digest}" if digest else tag

def validate_agent_image(image: str) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            DOCKER_PLATFORM,
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            (
                'set -eu; '
                'command -v opencode; '
                'opencode --version; '
                'test -n "$REPO_DIR"; '
                'test -d "$REPO_DIR/.git"; '
                'printf "REPO_DIR=%s\\n" "$REPO_DIR"'
            ),
        ],
        timeout=120,
        check=False,
    )

def agent_image_tag(
    baseline_tag: str,
    baseline_digest: str,
    dockerfile: Path,
) -> str:
    identity = json.dumps(
        {
            "baseline": baseline_digest or baseline_tag,
            "dockerfile_sha256": hashlib.sha256(
                dockerfile.read_bytes()
            ).hexdigest(),
        },
        sort_keys=True,
    )

    suffix = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return f"swe-pro-opencode:{suffix}"


def ensure_agent_image(
    *,
    baseline_tag: str,
    baseline_digest: str,
    pinned_baseline: str,
    dockerfiles: Sequence[Path],
    build_context: Path,
    force_rebuild: bool,
) -> str:
    """Reuse or build an image, trying the supplied Dockerfiles in order."""
    candidates = list(dockerfiles)
    if not candidates:
        raise ValueError("At least one agent Dockerfile is required")

    failures: List[str] = []
    baseline_pulled = False
    for candidate in candidates:
        if not candidate.is_file():
            failures.append(f"Dockerfile not found: {candidate}")
            continue

        derived = agent_image_tag(baseline_tag, baseline_digest, candidate)
        exists = run_command(
            ["docker", "image", "inspect", derived], check=False,
        ).returncode == 0
        if exists and not force_rebuild:
            validation = validate_agent_image(derived)
            if validation.returncode == 0:
                logger.info("Reusing validated agent image %s from %s\n%s",
                            derived, candidate, validation.stdout.strip())
                return derived
            logger.warning("Existing agent image %s is invalid; rebuilding:\n%s",
                           derived, validation.stdout.strip())
            run_command(["docker", "image", "rm", "--force", derived], check=False)

        if not baseline_pulled:
            logger.info("Pulling pinned baseline %s", pinned_baseline)
            run_command(
                ["docker", "pull", "--platform", DOCKER_PLATFORM, pinned_baseline],
                timeout=3600,
            )
            baseline_pulled = True

        command = [
            "docker", "build", "--platform", DOCKER_PLATFORM,
            "--file", str(candidate), "--tag", derived,
            "--build-arg", f"BASE_IMAGE={pinned_baseline}",
        ]
        if force_rebuild:
            command.append("--no-cache")
        command.append(str(build_context))

        logger.info("Building agent image %s using %s", derived, candidate)
        build = run_command(command, timeout=3600, check=False)
        if build.returncode != 0:
            details = f"{candidate}: build exited with {build.returncode}\n{build.stdout}"
            failures.append(details)
            logger.warning("Agent image build failed:\n%s", details)
            if candidate != candidates[-1]:
                logger.info("Trying the next Dockerfile candidate")
            continue

        validation = validate_agent_image(derived)
        if validation.returncode != 0:
            logger.error("Built agent image failed validation; keeping %s for inspection:\n%s",
                         derived, validation.stdout)
            raise RuntimeError(
                f"Built agent image failed validation: {derived}\n{validation.stdout}"
            )
        logger.info("Validated agent image %s from %s\n%s",
                    derived, candidate, validation.stdout.strip())
        return derived

    raise RuntimeError("Agent image build attempts failed:\n" + "\n".join(failures))


def docker_exec(
    container: str,
    argv: Sequence[str],
    *,
    workdir: Optional[str] = None,
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command: List[str] = ["docker", "exec"]
    if workdir:
        command += ["--workdir", workdir]
    command.append(container)
    command.extend(argv)
    return run_command(command, timeout=timeout, check=check)


def selected_config_environment(config: Path, model: str) -> List[str]:
    """Validate the selected custom provider and collect its environment names."""

    settings = json.loads(config.read_text(encoding="utf-8"))
    provider, separator, model_id = model.partition("/")
    if not separator or not provider or not model_id:
        raise ValueError("--model must have the form PROVIDER/MODEL")
    enabled = settings.get("enabled_providers")
    if enabled is not None and provider not in enabled:
        raise ValueError(f"Provider {provider!r} is not enabled in the config")
    if provider in settings.get("disabled_providers", []):
        raise ValueError(f"Provider {provider!r} is disabled in the config")
    provider_config = settings.get("provider", {}).get(provider)
    if not isinstance(provider_config, dict) or model_id not in provider_config.get("models", {}):
        raise ValueError(f"Model {model!r} is not declared in the selected config")
    selected = {
        "options": provider_config.get("options", {}),
        "model": provider_config["models"][model_id],
    }
    names = sorted(set(re.findall(
        r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", json.dumps(selected)
    )))
    for name in names:
        if not os.getenv(name, "").strip():
            raise ValueError(f"Required environment variable is missing: {name}")
    return names


def start_container(
    *,
    image: str,
    config: Path,
    name: str,
    memory: str,
    cpus: str,
    pids_limit: int,
    model: str = "public/gpt-4o",
) -> None:
    cmd = [
        "docker", "run", "--detach", "--rm",
        "--platform", DOCKER_PLATFORM,
        "--name", name,
        "--init",
        "--security-opt=no-new-privileges:true",
        "--memory", memory,
        "--cpus", cpus,
        "--pids-limit", str(pids_limit),
        "--add-host", "host.docker.internal:host-gateway",
        "--mount",
        f"type=bind,src={config},dst=/etc/opencode/opencode.json,readonly",
        "--env", "OPENCODE_CONFIG=/etc/opencode/opencode.json",
        "--env", "OPENCODE_DISABLE_AUTOUPDATE=1",
    ]

    for name in selected_config_environment(config, model):
        cmd += ["--env", name]

    cmd += [
        "--entrypoint", "/bin/sh",
        image,
        "-c", "while :; do sleep 3600; done",
    ]

    run_command(cmd, timeout=120)


def verify_baseline_repo(
    container: str,
    repo_dir: str,
    baseline_sha: str,
) -> Set[str]:
    """Verify the immutable baseline image; do not mutate its prepared state."""
    actual_sha = docker_exec(
        container,
        ["git", "rev-parse", "HEAD"],
        workdir=repo_dir,
    ).stdout.strip()
    if actual_sha != baseline_sha:
        raise RuntimeError(
            "Baseline image SHA mismatch: "
            f"row.json expects {baseline_sha}, image contains {actual_sha}"
        )

    tracked_diff = docker_exec(
        container,
        ["git", "status", "--porcelain", "--untracked-files=no"],
        workdir=repo_dir,
    ).stdout
    if tracked_diff.strip():
        raise RuntimeError(
            f"Baseline image has tracked modifications:\n{tracked_diff}"
        )

    untracked = docker_exec(
        container,
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        workdir=repo_dir,
    ).stdout
    return {path for path in untracked.split("\0") if path}


def stream_opencode(
    *,
    container: str,
    repo_dir: str,
    prompt_file: Path,
    model: str,
    timeout: int,
) -> str:
    run_command(
        [
            "docker",
            "cp",
            str(prompt_file),
            f"{container}:/tmp/swe-pro-prompt.md",
        ]
    )

    command = [
        "docker",
        "exec",
        "--workdir",
        repo_dir,
        container,
        "/bin/sh",
        "-c",
        # Save the PID so the host runner can terminate OpenCode without
        # deleting the container or losing the working-tree patch.
        (
            'echo "$$" > /tmp/swe-pro-opencode.pid; '
            'exec opencode run '
            '--format json '
            '--print-logs '
            '--model "$1" '
            '"$(cat /tmp/swe-pro-prompt.md)"'
        ),
        "run-opencode",
        model,
    ]

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    captured: List[str] = []
    turn_finished = threading.Event()

    def consume() -> None:
        assert process.stdout is not None

        for line in process.stdout:
            captured.append(line)
            print(line, end="", flush=True)

            stripped = line.strip()

            # Primary signal: OpenCode logs this exactly once, when the whole
            # run is genuinely finished
            if "exiting loop" in stripped.lower():
                turn_finished.set()
                continue

            # Secondary signal (structured): a step_finish whose reason is a
            # terminal reason. On multi-step runs there are several step_finish
            # events
            if stripped.startswith("{"):
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "step_finish":
                    reason = event.get("part", {}).get("reason")
                    if reason in {"stop", "length", "content_filter", "error"}:
                        logger.info(
                            "OpenCode step_finish reason=%s", reason
                        )

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout
    finished_seen_at: Optional[float] = None
    forced_after_finish = False

    def kill_opencode(signal: str) -> None:
        # Guard with kill -0 so a recycled PID is not signalled.
        docker_exec(
            container,
            [
                "/bin/sh",
                "-c",
                (
                    'if [ -f /tmp/swe-pro-opencode.pid ]; then '
                    'pid="$(cat /tmp/swe-pro-opencode.pid)"; '
                    'if kill -0 "$pid" 2>/dev/null; then '
                    f'kill -{signal} "$pid" 2>/dev/null || true; '
                    'fi; '
                    "fi"
                ),
            ],
            check=False,
        )

    while process.poll() is None:
        now = time.monotonic()

        if turn_finished.is_set():
            if finished_seen_at is None:
                finished_seen_at = now
                logger.info(
                    "OpenCode finished its turn; waiting briefly for the "
                    "process to exit on its own"
                )

            if now - finished_seen_at >= 3:
                logger.warning(
                    "OpenCode did not exit after finishing; terminating the "
                    "process and preserving the container"
                )

                kill_opencode("TERM")
                forced_after_finish = True

                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_opencode("KILL")
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                break

        if now >= deadline:
            # A real timeout, not normal turn completion.
            kill_opencode("TERM")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            reader.join(timeout=5)
            raise TimeoutError(f"OpenCode timed out after {timeout} seconds")

        time.sleep(0.5)

    reader.join(timeout=5)

    transcript = "".join(captured)
    return_code = process.returncode

    recoverable_prefill_error = (
        bool(return_code)
        and ASSISTANT_PREFILL_ERROR in transcript.lower()
    )

    if recoverable_prefill_error:
        logger.warning(
            "OpenCode encountered an assistant-prefill rejection; checking "
            "whether any working-tree diff can be saved"
        )
        return transcript

    # SIGTERM after a recognized turn completion is normal.
    if return_code and not forced_after_finish:
        raise RuntimeError(
            f"OpenCode exited with code {return_code}\n{transcript}"
        )

    return transcript


def container_git_diff(
    container: str,
    repo_dir: str,
    baseline_untracked: Set[str],
) -> str:
    current_untracked_raw = docker_exec(
        container,
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        workdir=repo_dir,
    ).stdout
    current_untracked = {
        path for path in current_untracked_raw.split("\0") if path
    }
    new_files = sorted(current_untracked - baseline_untracked)
    if new_files:
        docker_exec(
            container,
            ["git", "add", "-N", "--", *new_files],
            workdir=repo_dir,
        )
    return docker_exec(
        container,
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        workdir=repo_dir,
    ).stdout


def write_outputs(
    inst_dir: Path,
    *,
    diff: str,
    row: Dict[str, Any],
    model: str,
    provider: str,
    seconds: float,
    dataset_path: str,
    transcript: str,
    prompt_text: str,
    baseline_image: str,
    agent_image: str,
    container_repo_dir: str,
) -> None:
    inst_dir.mkdir(parents=True, exist_ok=True)
    # Preserve Git's diff verbatim: trailing blank context lines are part of
    # the hunk and must not be stripped or replaced with a single newline.
    (inst_dir / "completion.txt").write_text(diff, encoding="utf-8")
    (inst_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
    (inst_dir / "agent_transcript.txt").write_text(transcript, encoding="utf-8")
    (inst_dir / "meta.json").write_text(
        json.dumps(
            {
                "scenario_id": get_scenario_id(row),
                "repo": row.get("repo"),
                "baseline_sha": row.get("baseline_sha"),
                "seconds": seconds,
                "provider": provider,
                "model": model,
                "temperature": None,
                "top_p": None,
                "prompt_dataset_path": dataset_path,
                "pipeline": "agentic",
                "agent": "opencode",
                "retrieval": "agent_dynamic",
                "context": "task_only_unaugmented",
                "max_agent_steps": MAX_AGENT_STEPS,
                "baseline_image": baseline_image,
                "agent_image": agent_image,
                "container_repo_dir": container_repo_dir,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def remove_docker_image(image: str) -> None:
    result = subprocess.run(
        ["docker", "image", "rm", "--force", image],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if result.returncode == 0:
        logger.info("Removed Docker image: %s", image)
    else:
        logger.warning(
            "Could not remove Docker image %s:\n%s",
            image,
            result.stdout.strip(),
        )


def process_one(
    *,
    row: Dict[str, Any],
    prompt_text: str,
    out_inst_dir: Path,
    model: str,
    provider: str,
    timeout: int,
    dataset_path: str,
    opencode_config: Path,
    agent_dockerfiles: Sequence[Path],
    build_context: Path,
    force_rebuild: bool,
    memory: str,
    cpus: str,
    pids_limit: int,
    cleanup_images: bool,
) -> None:
    scenario_id = get_scenario_id(row)
    if (out_inst_dir / "DONE").exists():
        logger.info("%s: already done", scenario_id)
        return

    baseline_sha = str(row.get("baseline_sha") or "").strip()
    if not baseline_sha:
        raise ValueError(f"{scenario_id}: missing baseline_sha")

    out_inst_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = out_inst_dir / "prompt.md"
    prompt_file.write_text(prompt_text, encoding="utf-8")
    container = f"swe-pro-agent-{uuid.uuid4().hex[:12]}"
    started = time.monotonic()

    agent_image: Optional[str] = None
    baseline_image: Optional[str] = None
    baseline_tag: Optional[str] = None

    try:
        tag, digest, baseline_image = get_baseline_image(row)
        baseline_tag = tag
        agent_image = ensure_agent_image(
            baseline_tag=tag,
            baseline_digest=digest,
            pinned_baseline=baseline_image,
            dockerfiles=agent_dockerfiles,
            build_context=build_context,
            force_rebuild=force_rebuild,
        )

        start_container(
            image=agent_image,
            config=opencode_config,
            name=container,
            memory=memory,
            cpus=cpus,
            pids_limit=pids_limit,
            model=model,
        )
        # Every prepared SWE-Opt image defines this explicitly, e.g.
        # REPO_DIR=/repos/pandas. No filesystem search is performed.
        repo_dir = docker_exec(
            container,
            ["/bin/sh", "-c", 'printf "%s" "$REPO_DIR"'],
        ).stdout.strip()
        if not repo_dir.startswith("/repos/"):
            raise RuntimeError(
                f"Baseline image has invalid or missing REPO_DIR: {repo_dir!r}"
            )
        docker_exec(
            container,
            ["/bin/sh", "-c", 'test -d "$1/.git"', "verify-repo-dir", repo_dir],
        )
        logger.info("%s: using %s in %s", scenario_id, baseline_image, repo_dir)
        baseline_untracked = verify_baseline_repo(
            container,
            repo_dir,
            baseline_sha,
        )
        transcript = stream_opencode(
            container=container,
            repo_dir=repo_dir,
            prompt_file=prompt_file,
            model=model,
            timeout=timeout,
        )
        (out_inst_dir / "agent_transcript.txt").write_text(
            transcript, encoding="utf-8"
        )
        diff = container_git_diff(container, repo_dir, baseline_untracked)
        if (
            ASSISTANT_PREFILL_ERROR in transcript.lower()
            and not diff.strip()
        ):
            raise RuntimeError(
                "OpenCode hit the recoverable assistant-prefill error but "
                "produced no working-tree diff"
            )

        seconds = time.monotonic() - started
        write_outputs(
            out_inst_dir,
            diff=diff,
            row=row,
            model=model,
            provider=provider,
            seconds=seconds,
            dataset_path=dataset_path,
            transcript=transcript,
            prompt_text=prompt_text,
            baseline_image=baseline_image,
            agent_image=agent_image,
            container_repo_dir=repo_dir,
        )
        if not diff.strip():
            (out_inst_dir / "EMPTY.txt").write_text("empty diff\n", encoding="utf-8")
            logger.warning("%s: empty patch", scenario_id)
        (out_inst_dir / "ERROR.txt").unlink(missing_ok=True)
        (out_inst_dir / "DONE").write_text("ok\n", encoding="utf-8")
    except Exception as exc:
        logger.exception("%s failed", scenario_id)
        (out_inst_dir / "ERROR.txt").write_text(f"{exc}\n", encoding="utf-8")
    finally:
        # Always stop and remove the per-scenario container.
        run_command(
            ["docker", "rm", "--force", container],
            check=False,
        )

        if cleanup_images and agent_image:
            remove_docker_image(agent_image)


def main() -> None:
    parser = ArgumentParser(
        prog="OpenCode patch generation in prepared baseline images"
    )
    parser.add_argument(
        "--prompts_run_dir",
        type=Path,
        required=True
    )
    parser.add_argument("--out_root", type=Path, default=Path("inference/agent"))
    parser.add_argument("--run_id")
    parser.add_argument("--model", help="PROVIDER/MODEL; defaults to the selected config's model")
    parser.add_argument("--provider", help="Inferred from --model; must match when supplied")
    parser.add_argument("--agent_timeout", type=int, default=7200)
    parser.add_argument("--sleep_between", type=float, default=0.0)
    parser.add_argument("--scenario_id")
    parser.add_argument("--max_instances", type=int)
    parser.add_argument(
        "--docker_opencode_config",
        type=Path,
        default=Path("opencode.docker.json"),
    )
    parser.add_argument(
        "--agent_dockerfiles",
        dest="agent_dockerfiles",
        type=Path,
        nargs="+",
        default=[
            Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.instance-opencode",
            Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.instance-opencode-legacy-debian",
        ],
        help="Ordered Dockerfile candidates; retry the next candidate when a build fails.",
    )
    parser.add_argument("--build_context", type=Path, default=Path("."))
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument(
        "--cleanup_images",
        action="store_true",
        help="Remove derived images after each scenario; default keeps the cache.",
    )
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--pids_limit", type=int, default=512)

    args = parser.parse_args()

    prompts_dir = args.prompts_run_dir.expanduser().resolve()
    config = args.docker_opencode_config.expanduser().resolve()
    dockerfiles = [path.expanduser().resolve() for path in args.agent_dockerfiles]
    build_context = args.build_context.expanduser().resolve()
    for label, path in (
        ("prompts directory", prompts_dir),
        ("OpenCode config", config),
        ("build context", build_context),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {label}: {path}")

    try:
        settings = json.loads(config.read_text(encoding="utf-8"))
        args.model = args.model or settings.get("model")
        if not isinstance(args.model, str):
            raise ValueError("Supply --model or set model in the selected config")
        selected_config_environment(config, args.model)
    except ValueError as exc:
        parser.error(str(exc))
    model_provider = args.model.partition("/")[0]
    if args.provider is not None and args.provider != model_provider:
        parser.error("--provider must match the provider prefix in --model")
    args.provider = model_provider

    meta_path = prompts_dir / "PROMPT_META.json"
    prompt_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dataset_meta = prompt_meta.get("source_meta") or {}
    dataset_path = dataset_meta.get("dataset_path") or prompt_meta.get("dataset_path") or ""
    dataset_id = dataset_meta.get("prompt_dataset_id", "")

    if args.run_id:
        run_id = args.run_id
    else:
        model_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.model).strip("-")
        run_id = f"task_only__opencode-{model_tag}__agentic"
        if dataset_id:
            run_id += f"__{dataset_id}"

    run_dir = args.out_root.expanduser().resolve() / safe_id(run_id)
    (run_dir / "instances").mkdir(parents=True, exist_ok=True)
    (run_dir / "INFERENCE_META.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "provider": args.provider,
                "model": args.model,
                "prompt_dataset_meta": dataset_meta,
                "pipeline": "agentic",
                "agent": "opencode",
                "retrieval": "agent_dynamic",
                "context": "task_only_unaugmented",
                "max_agent_steps": MAX_AGENT_STEPS,
                "container_source": "row.docker.baseline",
                "reference_image_exposed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    instances = prompts_dir / "instances"
    if not instances.is_dir():
        raise SystemExit(f"Missing instances directory: {instances}")

    processed = 0
    for scenario_dir in sorted(p for p in instances.iterdir() if p.is_dir()):
        row_file = scenario_dir / "row.json"
        prompt_file = scenario_dir / "prompt.md"
        if not row_file.exists() or not prompt_file.exists():
            continue
        row = json.loads(row_file.read_text(encoding="utf-8"))
        if args.scenario_id and get_scenario_id(row) != args.scenario_id:
            continue

        process_one(
            row=row,
            prompt_text=prompt_file.read_text(encoding="utf-8"),
            out_inst_dir=run_dir / "instances" / safe_id(get_scenario_id(row)),
            model=args.model,
            provider=args.provider,
            timeout=args.agent_timeout,
            dataset_path=dataset_path,
            opencode_config=config,
            agent_dockerfiles=dockerfiles,
            build_context=build_context,
            force_rebuild=args.force_rebuild,
            memory=args.memory,
            cpus=args.cpus,
            pids_limit=args.pids_limit,
            cleanup_images=args.cleanup_images,
        )
        processed += 1
        if args.max_instances is not None and processed >= args.max_instances:
            break
        if args.sleep_between:
            time.sleep(args.sleep_between)

    if args.scenario_id and processed == 0:
        raise SystemExit(f"Scenario not found: {args.scenario_id}")
    logger.info("done: processed %d scenario(s) under %s", processed, run_dir)


if __name__ == "__main__":
    main()