# EvaLoop extension: the anchor-leak ablation

A fork of [evalooop/EvaLooop](https://github.com/evalooop/EvaLooop) that makes
EvaLoop's cross-cycle anchor explicit and swappable, so the contribution of the
one piece of ground truth that re-enters the loop on every cycle can be measured
instead of assumed.

> **Public fork:** <https://github.com/FirefoxSRV/EvaLooop> (branch `anchor-ablation`)
> **Branch:** `anchor-ablation` · **Upstream base:** `evalooop/EvaLooop@ce22fe2` (branch `dev`)
> **Patch against upstream:** `evaloop-anchor-ablation.patch`
> **Upstream's own README** is preserved verbatim as
> [`README.upstream.md`](README.upstream.md) - it documents the framework itself,
> which this file does not duplicate.

---

## 1. What this extends, and why

EvaLoop feeds a model its own output: generate code from a description, test it,
summarize the code, and let that summary become the next description. The number
of cycles sustained is the robustness score (ASL).

One string in that loop never degrades. `CodeSummarizationTask.postprocess`
takes the **final assert line of the original dataset prompt** and appends it to
every summary, forever. It is there for a real reason - a summary does not fix
the function's name, so without it the tests could not call the generated code -
but it carries more than a name. It carries one concrete input together with its
**ground-truth expected output**, re-injected from outside the loop on every
single cycle, in a framework whose premise is that the model's own output is the
only source of perturbation.

Three measurements on the provided data package motivated the extension:

| Measurement | Value |
|---|---|
| Spearman(pass@1, ASL) across all 107 leaderboard models | **0.945** (Pearson R² = 0.81) |
| Loops that pass cycle 1 and then survive all 10 cycles | **~91%** (median model) |
| Mid-loop failures per model, out of 378 tasks | **median 26** |
| Cycles whose generated code contains an echoed module-level assert | **33.4%** of all 165,630 (0% gemini-2.0-flash, 99.5% gpt-5) |

ASL is close to rank-equivalent with one-shot accuracy, because the loop barely
perturbs anything - and a third of all cycles copy the leaked assertion straight
into the solution.

> **Research question.** How much of EvaLoop's measured robustness is the model
> sustaining its own specification, and how much is a re-injected ground-truth
> anchor holding the task still?

Hypotheses, metrics, sampling and the failure codebook were fixed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any run.

## 2. The extension

One loop, three anchors. Generation prompt, summarization prompt, wrapper shape,
failure condition and decoding settings are all held fixed; only the anchor line
changes.

| Arm | Anchor | Carries | Source |
|---|---|---|---|
| **A0** `assert` | `assert find_length("11000010001") == 6` | name, arity, inputs, **expected output** | original prompt, constant — **this is upstream** |
| **A1** `masked_assert` | `assert find_length('11000010001') == ...` | name, arity, inputs | original prompt, oracle masked by AST |
| **A2** `signature` | `def find_length(s):` | name, parameter names | **the model's own code**, re-read each cycle |

A1 − A0 isolates the leaked oracle. A2 − A1 isolates the remaining
ground-truth-derived argument values. A2 is the only arm in which nothing but
model output crosses the cycle boundary.

Two properties worth stating plainly:

* **Cycle 1 is byte-identical in all three arms** (same prompt, same greedy
  settings), so `pass@1` is equal across arms *by construction*, and any
  difference in ASL is attributable to the anchor alone. The call cache makes
  this real rather than notional: the later arms replay cycle 1 from disk.
* **Signature stability**, which the assignment requires any Option B design to
  solve, is handled by parsing the code that just *passed* and emitting the
  `def` line of the function named by the dataset's `entry_point`. That code
  passed the benchmark's tests, so a callable function with the right name and
  arity exists by construction.

### The confound that had to be controlled

A1's anchor contains `== ...`. A model that copies it into its answer raises
`AssertionError` and fails for a *formatting* reason, not an information one -
and echo rates range from 0% to 99.5% by model, so this would have quietly
turned "echoes a lot" into "looks fragile". Worse, an echoed assertion is part of
the code that gets summarized next, so a summary could restate the expected
output in prose and re-inject exactly the oracle the ablation removes.

So **module-level `assert` statements are stripped from generated code before
testing and before summarization, in every arm including the baseline.** Applying
it uniformly is what keeps the arms comparable. It is a deliberate deviation from
upstream, and `scripts/retest_package_code.py` measures its effect on published
verdicts directly.

## 3. What was modified upstream, and why

The extension is almost entirely new files. **Five existing source files were
touched, adding 43 lines and removing 7** - every change additive or a
lazy-import move, none altering existing behaviour:

| File | Change | +/− | Why |
|---|---|---|---|
| `evaloop/evaluation/cycle.py` | expose `task_input` in the `postprocess` context, in both the batch and sequential paths | +12 −1 | The signature anchor must derive from the code the model just produced; upstream's context carried only the cycle-start input. Existing tasks ignore the new key. |
| `evaloop/models/factory.py` | import backends on use; register `gemini`/`google` | +13 −4 | An API-only run should not need torch installed, and a local run should not need the Gemini SDK. Extends the lazy-import convention the repo already asserts for its CLI. |
| `evaloop/core/evaluator.py` | import `py_testing` (and so evalplus) inside the two methods that use it | +11 −2 | The generation stage talks to an API and never executes model output; it should not need the execution machinery at all. Same convention. |
| `evaloop/tasks/__init__.py` | register the two new task types | +4 −0 | Follows the existing registry convention. |
| `evaloop/tasks/base.py` | document the new context key | +3 −0 | It is part of the hook contract. |

Two non-source files also change: `.gitignore` gains the run artifacts
(`cache/`, `results/`, `.evalplus-cache/`), and `README.md` is this file, with
upstream's preserved verbatim as `README.upstream.md`.

Everything else is new: `evaloop/anchors/`, two task types, a Gemini backend, the
call cache, the sandboxed tester, the anchored evaluator, the scripts, and the
tests. The upstream test suite still passes unchanged alongside them (209 tests
total).

A one-line check of that claim:

```bash
git diff --stat upstream/dev...HEAD                      # totals
git diff upstream/dev...HEAD -- evaloop/                 # every source change
```

**Things deliberately *not* reimplemented.** `evaloop/analysis/metrics.py`
already implements `asl_pow`, `asl_bias`, `asl_semantic` and `pass_drop` exactly
as the leaderboard defines them, and `evaloop/analysis/similarity.py` already
holds the paper's byte-identical judge prompt, score parser, boundary-pair
definition and score cache. Both are reused as-is.

## 4. Reproducing every reported number

### Setup

```bash
git clone <your fork> && cd EvaLooop && git checkout anchor-ablation

# Generation stage: no evalplus, no torch, never executes model output.
python3 -m venv .venv-gen
.venv-gen/bin/pip install google-genai pyyaml tqdm pytest

# Testing stage: a Linux container (see "Why a container" below).
./docker/run.sh setup        # build, warm the EvalPlus cache, verify isolation

export GEMINI_API_KEY="..."  # or pass --use-vertex for Vertex AI
```

### Validation gates — run these before trusting anything

```bash
# 1. Our metrics reproduce the published leaderboard.
.venv-gen/bin/python scripts/validate_leaderboard.py --data-dir ../assignment2-data
#    -> 56/60 models exact. The other 4 are the multi-run models the data
#       package README documents (leaderboard averages 5 runs; loops/ ships exp_1).

# 2. The A0 arm *is* upstream's loop, checked against the recorded data.
.venv-gen/bin/python scripts/validate_anchors_on_package.py --data-dir ../assignment2-data
#    -> 165,630/165,630 recorded cycle transitions reproduced byte-for-byte.

# 3. Our tester reproduces published pass/fail verdicts, and assert-stripping
#    does not change them.
.venv-gen/bin/python scripts/retest_package_code.py --n 400

# 4. Unit and loop tests (no API, no code execution).
.venv-gen/bin/python -m pytest tests/ -q
```

### The experiment

```bash
# The 150-task subset (seed-fixed, stratified, with its own fidelity check).
.venv-gen/bin/python scripts/sample_tasks.py --data-dir ../assignment2-data --n 150 --seed 20251013
#    -> ranking fidelity Spearman rho = 0.9372 across the 60 package models

# Metadata the generation stage needs (two strings per task, no evalplus).
.venv-gen/bin/python scripts/export_dataset_meta.py --source package --data-dir ../assignment2-data
#    -> entry points confirmed against stored passing code: 367/378, 0 contradicted

# All models, all three arms. Models run in parallel, anchors in sequence.
./scripts/run_overnight.sh
```

### Analysis

```bash
P=.venv-gen/bin/python

$P scripts/summarize_runs.py --results-dir results --full   # summary.csv + metrics.csv
$P scripts/stats.py --results-dir results                   # paired CIs, Wilcoxon, model spread
$P scripts/emit_native_results.py --results-dir results --verify
#   also writes each run in the framework's native shape and checks it through
#   the framework's own ResultAnalyzer

$P scripts/cost_report.py --results-dir results             # usage, per-task cost
$P scripts/failure_sampler.py --results-dir results --per-arm 8
#   -> analysis/failures_to_code.csv (code these BEFORE opening failures_key.json)

$P scripts/judge_similarity.py --results-dir results --dry-run
#   boundary similarity; needs API access, so it is optional and not used for
#   any number in the report

$P scripts/build_report.py                                  # report.md -> report.pdf
./scripts/package_submission.sh submission.zip              # assemble + checklist
```

### Which command produces which reported number

| Number in the report | Command |
|---|---|
| ρ(pass@1, ASL) = 0.945, 91% survival, 33.4% echo | `validate_anchors_on_package.py` |
| 165,630 transitions reproduced | `validate_anchors_on_package.py` |
| 56/60 leaderboard models reproduced | `validate_leaderboard.py` |
| 117/120 tester agreement, 0/40 stripping changes | `retest_package_code.py --n 120` |
| Subset fidelity ρ = 0.937 | `sample_tasks.py` |
| Entry points 367/378 confirmed | `export_dataset_meta.py --source package` |
| ASL / pass@1 / mid-failure table | `summarize_runs.py --full` |
| ΔASL, CIs, Wilcoxon | `stats.py` |
| 88.4% cache hits, 760 calls, token counts | `cost_report.py` |
| Failure coding, 11/20 oracle-dependent | `failure_sampler.py` then `analysis/coding_summary.json` |

## 5. Models and settings

The Gemini 2.5 series - the only models with baselines in the data package - now
returns `404 ... no longer available to new users`, so no evaluated model has a
published baseline and **the A0 arm run here is the baseline**. That is what the
assignment requires for Option B anyway ("the same models on the same tasks under
the standard loop"), and it removes a threat the data package introduces: its
`prompts/task_prompts.md` documents a *different* summarization prompt from the
one its data shows (see §7), so a published baseline would not have been strictly
comparable.

All runs: greedy (`temperature=0`, `top_p=1.0`), `max_output_tokens=1024`,
`seed=20251013`, thinking set to `minimal`, max 10 cycles, one run per
configuration, 150 MBPP+ tasks.

Thinking is pinned deliberately. Current Gemini models think by default, which
changes both the output distribution and the cost; `gemini-3.x` rejects
`thinking_budget` and takes `thinking_level` instead, and the client enforces
that only one of the two is set.

## 6. Why a container, and what it guarantees

Generated code is untrusted. The pipeline is split by trust level: the
**generation** stage talks to the API and never executes a line of model output;
the **testing** stage executes it inside a container with no network, as a
non-root user, with dropped capabilities, a read-only repo mount and
memory/CPU/pid caps. The test image carries no model SDK and is given no API key,
so it could not reach a provider even if the code it runs tried.

The container is also required for *correctness*, not only safety: EvalPlus's
`reliability_guard` sets `RLIMIT_AS`, which macOS rejects with `current limit
exceeds maximum limit`, so the MBPP+ suites cannot run natively on a Mac host at
all. `./docker/run.sh check` verifies each property.

Testing is batched per cycle, so a 10-cycle run costs 10 container launches
rather than one per task per cycle.

### Resumability and cost control

Every successful model call is cached to `cache/<model>.jsonl`, keyed by the
prompt and every decoding setting that can change the output. Completions are
written as they arrive, not per batch, so an interrupted run resumes from the
exact cycle it stopped at and re-analysis costs nothing. Failures are never
cached.

A call that exhausts its retry budget - essentially always sustained throttling -
**aborts the run** rather than returning an error string that would be tested as
if it were code, failed, and recorded as the model failing to sustain the loop.
That would bias ASL downward for reasons unrelated to the model or the anchor,
and nothing in the output would have said so.

## 7. Two discrepancies in the provided material

Both are reported in the write-up; neither is a defect in the fork.

1. **The data package's documented summarization prompt is not the one used.**
   `prompts/task_prompts.md` specifies "Summarize what the following code does in
   a detailed description". But 8,839 of the sampled summaries open with "write a
   python function to" and the median summary is ~28 words - one sentence. The
   prompt actually used is the one-sentence variant in the framework's own
   `core/config.py` and in `experiments/generation_summarization.yaml`. **This
   fork uses the latter**; the former would not reproduce the published data.
2. **`passDrop` means two different things.** The leaderboard column is a raw
   *task count* (pass@M − pass@1); the framework's `metrics.pass_drop` returns a
   *fraction of T*. Comparing a fresh run against the published column without
   noticing is wrong by a factor of T/100. `scripts/validate_leaderboard.py`
   compares against the count and says so.

## 8. Deliverables map

| Deliverable | Where |
|---|---|
| Report (≤3 pages) | `report.pdf` |
| Public fork link | top of this file |
| Patch against upstream | `evaloop-anchor-ablation.patch` |
| Results JSON per (model, configuration) | `results/<model>__<anchor>.json`, in the data package's schema |
| `summary.csv` | `results/summary.csv` (`model,configuration,n_tasks,success_rate,asl_score`) |
| Pre-registration | `PREREGISTRATION.md` |
| Manual failure coding | `analysis/failures_to_code.csv`, `analysis/failures_key.json` |

Per the assignment, the data package itself is **not** redistributed here;
records are referenced by `model_id`, `task_id` and `cycle`.
