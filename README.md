# SWE-Opt

> An evaluation harness for software engineering agents — run, benchmark, and compare LLM-powered coding agents against parametrized performance test in isolated Docker containers.
> The dataset used in this study is available online at:
[SWE-Opt Dataset]((https://huggingface.co/datasets/SWE-Opt/SWE-Opt))

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Supported Providers](#supported-providers)

---

## Overview

SWE_Opt provides a reproducible, containerised evaluation framework for testing and comparing software engineering agents. Each scenario runs inside an isolated Docker container, ensuring consistent, unbiased results across runs and environments.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | `>=3.10, <3.12` (3.11 recommended) |
| Docker | Latest stable |
| Git | Any recent version |

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd SWE-Opt
```

### 2. Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the following commands from the repository root.

### 3. Reproduce the locked Python environment

```bash
uv python install
uv sync --locked --extra notebook
uv run --locked python --version
```

The committed `.python-version` selects Python **3.11.14**. `pyproject.toml` declares dependencies and `uv.lock` records exact resolved versions. `uv sync --locked` creates `.venv`, installs the project in editable mode, and refuses to update an outdated lockfile. No environment activation is needed when using `uv run`.

For the OpenCode agentic workflow, the default dependencies are sufficient; OpenCode runs inside Docker. Install optional host features only when needed:

```bash
uv sync --locked --extra notebook --extra openai
uv sync --locked --extra bm25
```

Other extras are `anthropic`, `gemini`, `huggingface`, and `zhipu`. The `dev` dependency group is installed by default; use `--no-dev` for a runtime-only environment. BM25 also needs the Java runtime required by the selected Pyserini release.

Launch the quick-start notebook with the same environment:

```bash
python --version   # Python 3.11.x
which python       # .../SWE-Opt/.venv/bin/python
```

Run command-line modules through uv, for example:

```bash
uv run --locked python -m swe_opt.inference.generate_agentic_patches --help
```

When deliberately changing dependencies, update `pyproject.toml`, run `uv lock`, review the changes, and commit both files. Use `uv lock --upgrade` only for intentional dependency upgrades.

**Reproducibility boundary:** this lockfile controls the host Python environment, not benchmark dependencies inside Docker. Keep baseline image digests, dataset versions, model/config settings, prompts, and seeds fixed as well. The instance Dockerfiles currently download OpenCode from its installer; pinning the OpenCode version is a separate step for full agent reproducibility. External model responses and performance measurements can still vary.

### 4. Install and verify Docker

SWE-Opt runs evaluation scenarios inside Docker containers. Ensure Docker is installed and the daemon is running:

```bash
docker --version
docker compose version
docker run hello-world
```

If Docker is not installed, follow the official installation guide for [Docker Desktop](https://docs.docker.com/desktop/) or [Docker Engine](https://docs.docker.com/engine/install/).

---

## Usage

### Run the task-only OpenCode agent

The agentic workflow builds a task-only prompt for each benchmark row, then runs OpenCode in a Docker container based on that row's baseline image. It does not require a local dataset download: `--dataset_path` accepts a Hugging Face dataset ID or a local JSON/JSONL file. Hugging Face datasets use the `test` split.

First, start Docker and configure the provider/model in `opencode.docker.json`. The checked-in example uses an OpenAI-compatible provider; set its API key in your shell or project `.env`. Do not place credentials in notebooks or commit them.

Build prompts from the public dataset:

```bash
uv run --locked python -m swe_opt.prep.prompt.build_task_only_prompts \
  --dataset_path SWE-Opt/SWE-Opt \
  --out_root data/prompt \
  --run_id quickstart_agentic
```

The builder prints the prompt run directory under `data/prompt/`. Use that exact directory (it contains `PROMPT_META.json` and `instances/`; the name has a short hash suffix) in the inference command:

```bash
uv run --locked python -m swe_opt.inference.generate_agentic_patches \
  --prompts_run_dir "data/prompt/<printed-prompt-run-directory>" \
  --docker_opencode_config opencode.docker.json \
  --out_root inference/agent \
  --run_id quickstart_agentic \
  --max_instances 1
```

`--max_instances 1` runs a smoke test; remove it to process the full prompt run. The inference runner prints its output directory under `inference/agent/` (the directory name also has a short hash suffix). Pass that exact directory to `swe_opt.harness.run_evaluation`. Use a new `--run_id` when changing prompts, models, or other run settings; the prompt builder refuses to overwrite an existing prompt run.

The prompt builder's `--help` lists its dataset, output-root, and run-ID arguments. The inference runner's `--help` lists model, scenario, timeout, and container resource options. Both commands should be run from the repository root.

## Configuration

Create a `.env` file in the project root. You only need to add keys for the providers and data sources you intend to use — leave out anything that does not apply to your setup.

```bash
# LLM providers
OPENAI_API_KEY=your_key_here        # openai_chat, openai_responses
ANTHROPIC_API_KEY=your_key_here     # anthropic
GEMINI_API_KEY=your_key_here        # gemini
NVIDIA_API_KEY=your_key_here        # nvidia_nim
ZHIPU_API_KEY=your_key_here         # zhipu
OLLAMA_API_KEY=your_key_here        # ollama (local instance)

# Hugging Face
HUGGINGFACE_API_KEY=your_key_here   # hf provider (inference)
HUGGINGFACE_TOKEN=your_token_here   # dataset access — optional for public, required for gated/private

# Retrieval
GITHUB_TOKEN=your_token_here        # required for BM25 retriever (fetches repository data)
```

> **Security notice:** Never commit your `.env` file. Ensure it is listed in `.gitignore` before pushing any changes.

---

## Supported Providers

Only add the API key for the provider you are using. Ollama runs locally and does not require an external API key.

| Provider | Environment Variable | Notes |
|---|---|---|
| `openai_chat` | `OPENAI_API_KEY` | Chat completions endpoint |
| `openai_responses` | `OPENAI_API_KEY` | Responses API endpoint |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| `gemini` | `GEMINI_API_KEY` | Google Gemini models |
| `nvidia_nim` | `NVIDIA_API_KEY` | NVIDIA NIM inference |
| `zhipu` | `ZHIPU_API_KEY` | Zhipu AI models |
| `hf` | `HUGGINGFACE_API_KEY` | Hugging Face inference API |

### Dataset & retrieval keys

| Component | Environment Variable | Notes |
|---|---|---|
| Hugging Face datasets | `HUGGINGFACE_TOKEN` | Optional for public datasets; required for gated or private datasets |
| BM25 retriever | `GITHUB_TOKEN` | GitHub personal access token; required to fetch repository data |

---

## License
This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
