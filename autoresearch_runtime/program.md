# autoresearch

This workspace is the agent-facing runtime for autonomous training experiments.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date, such as `mar12`. The branch `autoresearch/<tag>` must not already exist.
2. Create the branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files for full context:
  - `prepare.py` — fixed constants, tokenizer, dataloader, evaluation. Do not modify.
  - `train.py` — the only file you modify.
  - `modal_runner.py` — the only command surface for running experiments. Do not modify.
  - `pyproject.toml` — available runtime dependencies. Do not modify.
4. Verify the Modal cache exists. If data has not been prepared yet, tell the human that one-time remote data preparation must be run from the operator workspace before experimentation starts.
5. Initialize `results.tsv` in the current directory with just the header row. The baseline will be recorded after the first run.
6. Confirm setup and start the loop.

## Execution contract

Each experiment runs on a single Modal `L40S` GPU for a fixed 5-minute training budget.

- `python modal_runner.py train > run.log 2>&1`

This is the only experiment execution command. Always redirect to `run.log` so the result can be parsed after the run. Do not run `uv` directly for experiments.

## Scratch work

You may use local Python for quick calculations, small hypothesis checks, or one-off scripts while reasoning.

- Use the operator workspace Python from the parent project: `uv run --project .. python`
- Example: `uv run --project .. python - <<'PY'`
- Do not use the runtime workspace Python for scratch work. The runtime project is CUDA-pinned for remote execution and may not work locally.
- Do not try to inspect anything in the parent project. Inspecting will incur a substantial damage to both you and the human.

## Allowed changes

- You may modify `train.py` only.
- Everything inside `train.py` is fair game: architecture, optimizer, hyperparameters, batch size, model size, training loop details.

## Forbidden changes

- Do not modify `prepare.py`.
- Do not modify `modal_runner.py`.
- Do not install packages or edit `pyproject.toml`.
- Do not modify the evaluation harness in `prepare.py`. `evaluate_bpb` is the ground-truth metric.
- Again, do NOT try to inspect anything in the parent project. Inspecting will incur a substantial damage to both you and the human.

## Goal

Minimize `val_bpb`. Lower is better.

The first run must always be the unmodified baseline through `python modal_runner.py train > run.log 2>&1`.

VRAM is a soft constraint: some increase is acceptable for a real gain, but avoid wasteful blowups.

Prefer simpler changes when the metric impact is similar. Deleting complexity for equal or better results is a win.

## Output format

Each run ends with a summary block like:

```text
---
val_bpb:          1.121406
training_seconds: 300.2
total_seconds:    377.0
peak_vram_mb:     22805.5
mfu_percent:      31.37
total_tokens_M:   147.8
num_steps:        282
num_params_M:     50.3
depth:            8
```

Use:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

to extract the key result lines from a finished run.

## Logging

Record every experiment in `results.tsv` as tab-separated values with this header:

```text
commit	val_bpb	memory_gb	status	description
```

- `commit`: short git hash
- `val_bpb`: metric, or `0.000000` for crashes
- `memory_gb`: `peak_vram_mb / 1024`, rounded to one decimal, or `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short description of the experiment

Do not commit `results.tsv`.

## Loop

Loop indefinitely once setup is complete:

1. Check the current git state.
2. Edit `train.py` with one concrete idea. Think deeply and mathematically. You can create scratch files and use python (as described earlier) for scratch works.
3. Commit the change.
4. Run `python modal_runner.py train > run.log 2>&1`.
5. Read `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If grep is empty, inspect `tail -n 50 run.log`. Fix obvious mistakes and retry a small number of times. If the idea is broken, mark it as a crash and move on.
7. Append the result to `results.tsv`.
8. Keep the commit only if `val_bpb` improved.
9. If the result is equal, worse, or crashed, revert to the previous good commit.

Each run should finish in about 5 minutes plus startup and evaluation overhead. If a run exceeds 10 minutes, kill it and treat it as a failure.

Once the loop starts, do not stop to ask the human whether to continue. Keep iterating until interrupted.