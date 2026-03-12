# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and, ideally, a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat).

## Layout

The repo is split into two UV projects:

- `autoresearch_runtime/` is the agent-facing runtime workspace.
- the repo root is the operator workspace for Modal execution, analysis, and documentation.

Inside `autoresearch_runtime/`:

- `prepare.py` contains fixed constants, data prep, tokenizer training, the dataloader, and evaluation.
- `train.py` is the only file the research agent should modify.
- `program.md` is the runtime-specific instruction file you point the agent at.
- `modal_runner.py` is the stable command surface the agent uses for remote runs.

At the repo root:

- `modal_app.py` defines the Modal image and the remote CPU/GPU functions.
- `modal_runner.py` is the operator wrapper around `modal run`.
- `analysis.ipynb` is a local notebook for analyzing results.
- `pyproject.toml` contains only operator dependencies such as Modal and plotting libraries.

By design, training runs for a fixed 5-minute budget, excluding startup and compilation. The metric is `val_bpb` and lower is better.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) is a good place to start.

## Local GPU quick start

Requirements: a single NVIDIA GPU and Python 3.10+.

```bash
# 1. Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Enter the runtime workspace and install runtime deps
cd autoresearch_runtime
uv sync

# 3. Download data and train tokenizer
uv run prepare.py

# 4. Run one 5-minute experiment
uv run train.py
```

## Modal quick start

This repo also supports a local-agent / remote-execution workflow on [Modal](https://modal.com/). In this setup:

- `autoresearch_runtime/prepare.py` runs once on Modal CPU and populates a persistent volume.
- `autoresearch_runtime/train.py` runs on a single Modal `L40S`.
- the agent stays in `autoresearch_runtime/` and uses `python modal_runner.py train`.

```bash
# 1. Install the Modal CLI outside the runtime environment
uv tool install modal

# 2. Authenticate once
modal setup

# 3. Install the operator workspace deps
uv sync

# 4. Run one-time remote data prep
python modal_runner.py prepare

# 5. Run one remote training experiment
python modal_runner.py train
```

Notes:

- `python modal_runner.py train` streams live logs and ends with the same summary block as a direct local run.
- the remote cache lives in a persistent Modal volume, so re-running `prepare` is additive.
- the currently validated cloud setup uses one Modal `L40S`.

## Running the agent

Point the agent at [`autoresearch_runtime/program.md`](autoresearch_runtime/program.md).

A typical prompt is:

```text
Read program.md and let's kick off a new experiment. Let's do the setup first.
```

For Modal-backed runs, the agent should work inside `autoresearch_runtime/` and use:

```bash
python modal_runner.py train > run.log 2>&1
```

That wrapper keeps the Modal details outside the agent-facing workspace.

## Project structure

```text
autoresearch_runtime/             agent-facing runtime UV project
autoresearch_runtime/prepare.py   fixed data prep + runtime utilities
autoresearch_runtime/train.py     model, optimizer, training loop
autoresearch_runtime/program.md   agent instructions
autoresearch_runtime/modal_runner.py
autoresearch_runtime/pyproject.toml
modal_app.py                      Modal image + remote execution functions
modal_runner.py                   repo-root wrapper around `modal run`
pyproject.toml                    operator dependencies
analysis.ipynb                    local analysis notebook
```

## Design choices

- Single file to modify. The agent only touches `autoresearch_runtime/train.py`.
- Fixed time budget. Every experiment gets the same 5-minute training window.
- Clear boundary. Runtime code and operator tooling live in separate UV projects.
- Self-contained loop. One GPU, one metric, one file to change.

## Platform notes

This code currently assumes a single NVIDIA GPU. For cloud execution, the included Modal path keeps the coding agent local while running `autoresearch_runtime/prepare.py` and `autoresearch_runtime/train.py` remotely.

If you want to target much smaller machines, the main knobs are:

1. Use a lower-entropy dataset such as [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
2. Lower `VOCAB_SIZE`, `MAX_SEQ_LEN`, and `EVAL_TOKENS` in `autoresearch_runtime/prepare.py`.
3. Lower `DEPTH` and `TOTAL_BATCH_SIZE` in `autoresearch_runtime/train.py`.
4. Prefer simpler attention patterns such as `"L"` if bandwidth is tight.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
