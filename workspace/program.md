# autoresearch — nanogpt

This repo is a self-contained task for autonomous training experiments.

Your workspace is `workspace/`. Task execution is owned by `.runner/` and is invoked from the task repo root via `python3 run.py ...`. Experiment tracking is handled separately by the shared `autoresearch` CLI.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date, such as `mar17`. The branch `<tag>` must not already exist.
2. Start in an isolated worktree. If the user has not already created one, from the task repo root run:
   ```bash
   git worktree add .worktrees/<tag> -b <tag>
   ```
   Then work inside `.worktrees/<tag>/`. If you are already in a worktree, continue there.
3. Read the in-scope files for full context:
  - `prepare.py` — fixed constants, tokenizer, dataloader, evaluation. Do not modify.
  - `train.py` — the only code file you modify.
  - `exp.md` — create for each experiment with hypothesis and reasoning.
  - `.runner/modal/pyproject.toml` from the repo root — remote runtime manifest. Read-only context.
4. Verify the Modal cache exists. If data has not been prepared yet, from the task repo root run:
   ```bash
   python3 run.py prepare
   ```
5. Confirm setup and start the loop.

Working in a dedicated worktree keeps your changes isolated from other agents and avoids git conflicts when running parallel experiments.

## Execution contract

Each experiment runs on a single Modal `L40S` GPU for a fixed 5-minute training budget.

```bash
python3 run.py train > run.log 2>&1
```

This is the only experiment execution command. Always redirect to `run.log` so the result can be parsed after the run.

## Scratch work

You may use local Python for quick calculations or small hypothesis checks:

```bash
python - <<'PY'
import math
print(math.sqrt(2))
PY
```

## Allowed changes

- You may modify `train.py` only (and create `exp.md` per experiment).
- Everything inside `train.py` is fair game: architecture, optimizer, hyperparameters, batch size, model size, training loop details.

## Forbidden changes

- Do not modify `prepare.py`, `run.py`, or anything under `.runner/`.
- Do not install packages.
- Do not modify the evaluation harness in `prepare.py`. `evaluate_bpb` is the ground-truth metric.

## Goal

Minimize `val_bpb`. Lower is better.

The first run must always be the unmodified baseline:

```bash
python3 run.py train > run.log 2>&1
```

VRAM is a soft constraint: some increase is acceptable for a real gain, but avoid wasteful blowups.

Prefer simpler changes when the metric impact is similar. Deleting complexity for equal or better results is a win.

## Output format

Each run ends with a summary block:

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

Extract key results:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## exp.md

Create `exp.md` for each experiment. Write it before the run commit. Include:

- **Hypothesis**: what you expect and why
- **Reasoning**: mathematical derivation, parameter calculations, or conceptual argument
- **References**: papers and what you took from them
- **Changes**: which file and what changed
- **Base**: parent commit hash and baseline val_bpb

After the run, append a **Results** section with val_bpb, peak_vram_mb, status (keep/discard/crash), and brief analysis. Then make a second commit with the completed results note.

## Experiment recording

Record each experiment once, after the results commit exists. Each DB row stores both the run commit and the later results commit:

```bash
autoresearch record <run-commit> \
  --result-commit <result-commit> \
  --status success \
  --decision keep|discard \
  --description "one-line summary" \
  --metric val_bpb=<value> \
  --metric peak_vram_mb=<value>
```

For NanoGPT, extract the numeric fields from the final summary block in `run.log` and pass them explicitly as repeated `--metric name=value` flags. Include all numeric summary fields, not just `val_bpb`.

Browse experiments:

```bash
autoresearch summary
autoresearch read <commit-hash>
```

## Loop

Loop indefinitely once setup is complete:

1. Query `autoresearch summary` and read past exp.md via `autoresearch read <commit>` to understand what has been tried.
2. Check the current git state.
3. Edit `train.py` with one concrete idea. Write `exp.md` with hypothesis and reasoning. Think deeply and mathematically.
4. Commit the runnable snapshot. Save the hash as the run commit.
5. Run: `python3 run.py train > run.log 2>&1`.
6. Check: `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
7. If grep is empty, inspect `tail -n 50 run.log`. Fix obvious mistakes and retry a small number of times. If the idea is broken, write the failure into `exp.md`, commit the results note, and record with `--status crash` or `--status timeout`.
8. Append Results to `exp.md`, including the keep/discard decision. Commit that update. Save the hash as the result commit.
9. Record once: `autoresearch record <run-commit> --result-commit <result-commit> --status success --decision keep|discard --description "..." --metric val_bpb=<...> --metric peak_vram_mb=<...> ...`.
10. Keep the result commit only if `val_bpb` improved. If equal, worse, or crashed, revert to the previous good result commit.

Each run should finish in about 5 minutes plus startup and evaluation overhead. If a run exceeds 10 minutes, kill it and treat it as a failure.

Once the loop starts, do not stop to ask the human whether to continue. Keep iterating until interrupted.
