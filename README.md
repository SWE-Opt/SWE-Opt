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

### 2. Create a Python 3.11 virtual environment

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
```

Verify the environment:

```bash
python --version   # Python 3.11.x
which python       # .../SWE-Opt/.venv/bin/python
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

### 4. Install and verify Docker

SWE-Opt runs evaluation scenarios inside Docker containers. Ensure Docker is installed and the daemon is running:

```bash
docker --version
docker compose version
docker run hello-world
```

If Docker is not installed, follow the official installation guide for [Docker Desktop](https://docs.docker.com/desktop/) or [Docker Engine](https://docs.docker.com/engine/install/).

---

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
