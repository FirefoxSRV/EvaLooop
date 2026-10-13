# Pre-registration: the anchor-leak ablation

Committed before any generation run. Hypotheses, metrics, sampling and the
failure codebook are fixed here so the analysis cannot be fitted to the results
after the fact.

## Research question

EvaLoop measures robustness as the number of generation <-> summarization cycles
a model sustains. Across the 107 models on the published leaderboard, that score
is close to rank-equivalent with one-shot accuracy (Spearman(pass@1, ASL) =
0.945, Pearson R^2 = 0.81), and about 91% of loops that pass cycle 1 never fail
within ten cycles. The loop barely perturbs anything.

One string in the loop never degrades: the assert line of the *original* dataset
prompt, re-appended verbatim to every summary. It carries the function name, the
arity, one concrete input and that input's ground-truth expected output, and it
re-enters the loop from outside on every cycle.

> **RQ.** How much of EvaLoop's measured robustness is the model sustaining its
> own specification, and how much is a re-injected ground-truth anchor holding
> the task still?

## Configurations

One loop, one pluggable anchor. Everything else - generation prompt,
summarization prompt, wrapper shape, failure condition, decoding settings - is
held fixed.

| id | anchor | carries | source |
|----|--------|---------|--------|
| A0 | `assert` | name, arity, inputs, **expected output** | original prompt, constant (= upstream) |
| A1 | `masked_assert` | name, arity, inputs | original prompt, oracle replaced by `...` |
| A2 | `signature` | name, parameter names | the model's own code, re-read each cycle |

A1 - A0 isolates the leaked oracle. A2 - A1 isolates the remaining
ground-truth-derived argument values. A2 is the only arm where nothing but model
output crosses the cycle boundary.

Cycle 1 is byte-identical in all three arms, so pass@1 is equal by construction
and any difference in ASL is attributable to the anchor alone.

## Hypotheses

* **H1 (leak).** ASL falls monotonically A0 > A1 > A2 and mid-loop failures rise
  sharply. Prediction: the A0 -> A2 drop in ASL exceeds 1.0 loop, and the paired
  Wilcoxon test on per-task sustained cycles rejects at alpha = 0.05.
* **H2 (genuine robustness).** ASL is flat across anchors (|dASL| < 0.3 and no
  significant paired difference). The 91% survival rate reflects real robustness
  and the anchor is doing no work.
* **H3 (drift vs. information loss).** If H1 holds, failures under A1/A2 are
  preceded by *less* semantic drift than under A0 - i.e. **higher** boundary
  similarity between the original prompt and the last summary before failure -
  because the cause is lost test information rather than a wandering
  specification. H3 is what separates the two mechanisms and is falsifiable
  independently of H1.
* **H4 (discriminative power).** Removing the oracle widens the spread of
  robustness scores across models: sd(ASL_bias) under A2 > under A0.

An outcome where A1 and A2 barely move is a real answer to the RQ, not a failed
experiment, and will be reported as such.

## Models

Three models, all of which have published single-run baselines in the data
package: `gemini-2.0-flash-lite`, `gemini-2.5-flash-lite`, `gemini-2.5-flash`.
Greedy decoding (temperature 0, top_p 1), max 1024 output tokens, thinking
disabled on the 2.5 series, max 10 cycles, one run per configuration.

## Tasks

150 MBPP+ tasks, drawn by `scripts/sample_tasks.py` with seed 20251013:
5 equal quantile strata over each task's mean sustained loops across the 60
package models, 30 drawn per stratum, from the 362 tasks whose assert states a
comparison (the other 16 assert bare truthiness and admit no masked anchor).
Realised ranking fidelity: Spearman rho = 0.9372 between subset and
full-benchmark ASL across the 60 package models.

## Metrics

Computed with the framework's own `evaloop.analysis.metrics`, validated against
the published leaderboard (56/60 models exact; the 4 others are the multi-run
models the data package README documents).

Primary: `asl_pow` (= leaderboard `aslScore`). Secondary: pass@1, `asl_bias`
(= `robustnessScore`), pass-drop as a task count, mid-loop failure count, and
the histogram of sustained cycles. Paired across arms on identical tasks:
bootstrap CI on dASL (10k task resamples) and Wilcoxon signed-rank on per-task
sustained cycles.

## Confound control, fixed in advance

Module-level `assert` statements are stripped from generated code before testing
and before summarization, in **every** arm including the baseline. Rationale: in
the published records 33.4% of all cycles contain an echoed module-level
assertion (gpt-5: 99.5%). Left in place, an echo would (a) raise AssertionError
under A1's `== ...` anchor, turning an information effect into a formatting
artefact, and (b) let a summary of the echoing code restate the expected output,
re-injecting through prose the oracle the ablation removed. The baseline is
reported with and without the rule so its effect is visible.

## Failure codebook

Fixed before coding. Each failure gets exactly one primary code.

| code | failure mode |
|------|--------------|
| C1 | summary drops or garbles a requirement (spec loss) |
| C2 | summary describes a different but coherent task (semantic drift) |
| C3 | summary is faithful, the code does not implement it |
| C4 | signature/anchor failure: wrong name, arity or parameter semantics |
| C5 | extraction or formatting failure (no code block, prose as code, truncation) |
| C6 | oracle dependence: survived longer under A0, and the failing code is wrong on the leaked example input |
| C7 | plus-only failure: base tests pass, extended EvalPlus tests do not |
| C8 | harness or other (timeout, exception, unparseable) |

Sample: 30 failures, 10 per arm, stratified by failure cycle (2-3 / 4-6 / 7-10),
seed-fixed. Coded blind - `scripts/failure_sampler.py` shuffles rows and holds
the arm labels in a separate key file. 10 re-coded after 48h, Cohen's kappa
reported.

## Deviations

Any departure from this document will be listed in the README with its reason.
