# The assert line is doing the work: an anchor-leak ablation of EvaLoop

**Shreyas Visweshwaran** · CSC 591/791 Trustworthy AI for Software Engineering · Assignment 2
Fork: <https://github.com/FirefoxSRV/EvaLooop> (branch `anchor-ablation`)

## 1. Extension and research question

EvaLoop measures robustness by feeding a model its own output: generate code from a description, test it, summarize the code, and let that summary become the next description. Three measurements on the provided data package suggested to me that the loop does less work than it appears to. Across all 107 leaderboard models I measured Spearman(pass@1, ASL) = **0.945** (R² = 0.81), so ASL is nearly rank-equivalent with one-shot accuracy; **91%** of loops passing cycle 1 never fail within ten cycles; and the median model has only **26** mid-loop failures out of 378.

One string never degrades. `CodeSummarizationTask.postprocess` appends the **final assert line of the original prompt** to every summary, forever. It exists for a real reason, since a summary does not fix the function's name and the tests could not otherwise call the code, but it carries more than a name: one concrete input with its **ground-truth expected output**, re-injected from outside the loop every cycle, in a framework whose premise is that model output is the only source of perturbation. Models exploit it: **33.4%** of all 165,630 recorded cycles contain an echoed module-level assertion (99.5% for gpt-5, 0% for gemini-2.0-flash).

> **RQ.** How much of EvaLoop's measured robustness is the model sustaining its own specification, and how much is a re-injected ground-truth anchor holding the task still?

I expected removing the oracle to lower ASL and raise mid-loop failures (H1). Hypotheses were fixed in `PREREGISTRATION.md` before any run.

## 2. The framework and my changes

**The pipeline.** `EvaLoopEvaluator` loads the 378 MBPP+ tasks, seeding each with the benchmark's docstring prompt. Per cycle, `BatchEvaluationCycle` runs the configured tasks through one interface, `create_prompt → model → extract_output → postprocess`. The generation task renders `"Generate Python code for the following task: …"` and extracts the first triple-backtick block. That code runs against MBPP+'s base and extended EvalPlus suites via `untrusted_check`, passing only if both pass. The summarization task then prompts for a one-sentence summary of that code, and its `postprocess` wraps the reply as a docstring with the assert line re-appended, which becomes the next cycle's description. A failing task stops; survivors continue to ten cycles. `successful_cycles` counts cycles before the first failure, and `analysis/metrics.py` turns that vector into ASL = Σᵢ nᵢ·i²/(T·M), whose i² weighting pays compound credit for sustained cycles.

**What I changed.** The anchor was the only thing I needed to vary, and the `dev` branch already routes it through `postprocess`, so the extension is almost entirely new files: `evaloop/anchors/` (a registry mirroring `tasks.factory`, plus three strategies), two new task types, a Gemini backend, a disk-backed call cache, a sandboxed batch tester, and an evaluator subclass. **I changed five existing source files, +43/−7 lines.** The one substantive edit is in `evaluation/cycle.py`: upstream's `postprocess` context carried only the cycle-start input, and a model-derived anchor needs the code that was *just produced*, so I made the context also carry `task_input` (in both the batch and sequential paths; existing tasks ignore it). The rest are lazy-import moves in `models/factory.py` and `core/evaluator.py` so the generation stage needs neither torch nor evalplus, plus two registration lines and a docstring. I reused `analysis/metrics.py` and `analysis/similarity.py` rather than reimplementing the metrics or the judge.

| Arm | Anchor | Carries |
|---|---|---|
| **A0** `assert` | `assert find_length("11000010001") == 6` | name, arity, inputs, **expected output**. This is upstream unchanged. |
| **A1** `masked_assert` | `assert find_length('11000010001') == ...` | name, arity, inputs; oracle masked by AST |
| **A2** `signature` | `def find_length(s):` | name and parameter names, parsed from the model's own code each cycle |

A1−A0 isolates the oracle, and A2−A1 isolates the remaining ground-truth-derived argument values. A2 is the only arm where nothing but model output crosses the boundary. For **signature stability**, the problem the assignment requires any Option B design to solve, I parse the code that just *passed* and emit the `def` line for the dataset's `entry_point`. Since that code passed the tests, a callable function with the right name and arity exists by construction. Because cycle 1 is byte-identical across arms and the call cache is shared per model, **pass@1 is equal across arms by construction**, so ΔASL is attributable to the anchor alone.

**One confound had to be controlled.** A1's anchor ends `== ...`, so a model that echoes it raises `AssertionError` and fails for a formatting reason, and an echoed assertion is part of the code summarized next, which could restate the oracle in prose. I therefore strip module-level `assert` statements from generated code before testing *and* before summarization, in **every** arm including the baseline.

**Validation before any claim.** With `anchor: assert` the new task reproduces **all 165,630** recorded cycle transitions in the data package byte-for-byte, so the baseline arm is provably the published loop. `analysis/metrics.py` reproduces the leaderboard for **56/60** models (the other four are the multi-run models its README documents). Re-executing 120 stored solutions through my sandbox reproduces the recorded verdict for **117/120 (97.5%)**, and the assert-stripping rule changed **0 of 40** verdicts on samples that contained a module-level assertion, so I am satisfied the rule is immaterial to outcomes, as the design requires. The framework's own `ResultAnalyzer`, run on my exported results, returns my ASL to ten decimal places. 209 tests pass.

## 3. Experimental setup

Three models via the Gemini API: `gemma-4-26b-a4b-it` and `gemma-4-31b-it` with all three arms, and `gemini-3.5-flash-lite`, from a different family, with **A0 and A2**, the arms that carry the headline contrast. Its A1 arm is absent because a 500-requests-per-day account cap ended an earlier attempt mid-run; the harness refused to write a partial result rather than record throttled calls as model failures, and the run resumed from cache on a fresh quota. The Gemini 2.5 series, the only models with baselines in the data package, now returns `404 … no longer available to new users`, so no evaluated model has a published baseline and **the A0 arm is the baseline**, which is what Option B requires anyway. 150 MBPP+ tasks, drawn by `scripts/sample_tasks.py` with seed 20251013: five equal quantile strata over each task's mean sustained cycles across the 60 package models, 30 per stratum, from the 362 tasks whose assert states a comparison (16 assert bare truthiness and admit no masked anchor). The subset reproduces the 60-model ASL ranking at Spearman **ρ = 0.937**. Greedy (`temperature=0`, `top_p=1`), `max_output_tokens=1024`, `seed=20251013`, thinking pinned to `minimal`, max 10 cycles, one run per configuration.

Untrusted code is executed only inside a container with `--network none`, a non-root user, dropped capabilities, a read-only repo mount and memory/CPU/pid caps; the test image holds no model SDK and no API key. The container is required rather than advisory: EvalPlus's `reliability_guard` sets `RLIMIT_AS`, which macOS rejects.

**Cost.** All runs were on the free tier: **$0.00 billed**. Across all eight runs: **1,909 API calls** and 292,697 tokens (176,086 in / 116,611 out), with **15,813 calls served from cache: 89.2% of every call the loop asked for was a byte-identical repeat.** One model's full three-arm ablation costs 760 calls, about 5 calls and 816 tokens per task-arm.

**LLM use beyond the models under evaluation.** I used an LLM as a classifier for the failure coding in Section 4, and I report it here as the assignment requires.

*Model.* Claude Opus 4, accessed through Claude Code. *Input per example.* Four artifacts drawn from the run records: the original MBPP+ task, the summary that fed the failing cycle, the code that passed the previous cycle, and the code that failed. The arm label was withheld, so the classification was blind to the condition under test. *Inference settings.* No per-example decoding parameters were set; this ran inside an interactive assistant session rather than as a metered API call, so temperature, token limits and seed were not controlled and the pass is not reproducible in the way the model runs are. *Conversion to a decision.* Exactly one primary code per failure, drawn from the eight-code codebook fixed in `PREREGISTRATION.md`, applying the rule that C1 covers the loss of a detail of the same operation and C2 a shift to a different coherent operation. C6 was excluded from the classifier's scope because it requires cross-arm information; `scripts/score_coding.py` derives it from sustained-cycle data instead. *Cost.* No separately metered per-example charge.

Two limitations follow and I would rather state them than let a reader infer them. There is **one coder, not two**, so the Cohen's kappa the codebook anticipated does not exist and the distribution in Section 4 has no inter-rater check behind it. And the 20 coded failures are close to the entire population of 24 mid-loop failures available, so this is nearer a census than a sample, which removes sampling error but also any claim to independence between the sample and the result.

The boundary-similarity judge (`scripts/judge_similarity.py`) reuses the framework's byte-identical prompt from the data package but I did **not** run it, so no number in this report depends on it. Had I run it, its scores would not have been comparable to the published `gpt-4-turbo-2024-04-09` ones.

## 4. Results and failure analysis

| model | arm | pass@1 | ASL | ASL_bias | mid-fail | surv. 10 | ΔASL [95% CI] | Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| gemma-4-26b | A0 `assert` | 0.747 | 7.207 | 9.652 | 4 | 108 | n/a | n/a |
| | A1 `masked_assert` | 0.747 | 6.943 | 9.298 | 8 | 104 | −0.264 [−0.59, +0.00] | p = 0.066 |
| | A2 `signature` | 0.747 | 6.681 | 8.947 | 12 | 100 | **−0.526 [−0.92, −0.13]** | **p = 0.020** |
| gemma-4-31b | A0 `assert` | 0.773 | 7.337 | 9.488 | 6 | 110 | n/a | n/a |
| | A1 `masked_assert` | 0.773 | 7.073 | 9.147 | 10 | 106 | −0.264 [−0.66, +0.13] | p = 0.227 |
| | A2 `signature` | 0.773 | 6.879 | 8.896 | 13 | 103 | −0.458 [−0.98, +0.00] | p = 0.110 |
| gemini-3.5-flash-lite | A0 `assert` | 0.773 | 7.277 | 9.410 | 7 | 109 | n/a | n/a |
| | A2 `signature` | 0.773 | 6.755 | 8.734 | 15 | 101 | **−0.523 [−0.98, −0.08]** | **p = 0.011** |

*(150 tasks each; paired bootstrap, 10k task resamples; two-sided Wilcoxon signed-rank on per-task sustained cycles.)*

**The effect replicates across families, at a consistent magnitude.** All three models lose ASL when the oracle is removed, by strikingly similar amounts (**−0.526, −0.458 and −0.523**), and I saw mid-loop failures roughly double or triple in every case (4→12, 6→13, 7→15). Two of the three A0→A2 differences are significant at 0.05 (p = 0.020 and p = 0.011); `gemma-4-31b` gives p = 0.110 with a confidence interval that grazes zero. That one null is a power limitation rather than a contradiction: the arms differ on only 11–16 of 150 tasks, so each paired test has little to work with. I report the models separately rather than pooling them to manufacture a single significant result. **H1 is supported in direction and replicated across two model families, at a magnitude well below the >1.0 loop I pre-registered.** The leak is load-bearing for a *minority* of tasks, not for robustness in general.

**Does the ranking change?** No. Under A0 the three models sit within 0.13 ASL of each other (7.207, 7.277, 7.337) despite spanning two families, and A2 preserves their order exactly (Spearman ρ = 1.000, n = 3). The spread of ASL_bias does not widen either, at 0.101 under A0 against 0.091 under A2, so removing the oracle did **not** make the models more distinguishable. That refutes pre-registered H4, which predicted it would, on the theory that the leak was masking real differences between models. It did not: the leak inflates every model's score by a similar amount, so taking it away moves them together. Three models is still a weak basis for a ranking claim, but it is the same conclusion the within-model tests reach: these models differ less in loop robustness than ASL's absolute spread suggests.

**What the extension reveals that the standard loop does not.** Two structural findings, neither of which ASL exposes.

**The sustained-cycle distribution is almost perfectly bimodal.** Under A0, 38 tasks fail at cycle 1, 4 fail at cycles 2–3, and 108 survive all ten; **nothing fails between cycles 4 and 9**. The entire ablation effect lands on "sustained exactly one cycle" (2 → 6 → 9). The oracle matters precisely at the first handoff from the model's own description.

**The loop reaches a fixed point almost immediately.** 108 of the 110 tasks that get past cycle 1 reach a *repeated* (code, summary) state, at median **cycle 2** under A0 and cycle 3 under A2. The model regenerates identical code, which yields an identical summary, indefinitely. This is what the 88.4% cache-hit rate is measuring. So cycles 4–10 re-derive a fixed point while ASL's i² weighting pays quadratic credit for them, and weakening the anchor measurably *delays* convergence.

**Manual inspection.** I coded 20 mid-loop failures blind against the pre-registered codebook. The sampler shuffles the rows and withholds the arm labels in a separate key file, because knowing that a failure came from an ablated arm would make it trivial to code as oracle dependence. I therefore did not code C6 at all; I derive it afterwards from the run data.

| arm | C1 detail loss | C2 operation drift | C4 signature | C5 extraction | C7 plus-only | n |
|---|---|---|---|---|---|---|
| A0 `assert` | 1 | 0 | 1 | 1 | 1 | 4 |
| A1 `masked_assert` | 4 | 3 | 1 | 0 | 0 | 8 |
| A2 `signature` | 4 | 4 | 0 | 0 | 0 | 8 |

The modes differ sharply. The baseline failures I read were mechanical and idiosyncratic: one nested code fence that truncated mid-function, one `(m, n)` parameter inversion, one extended-test edge case, one garbled recursion. The ablated failures are **almost entirely specification degradation**, 8 of 8 and 8 of 8 in C1 plus C2.

Independent of how the mechanisms are coded, the cross-arm arithmetic is fixed: **11 of the 20 sampled failures are oracle-dependent**, meaning the same task sustained strictly longer under A0, and **10 of those went from surviving all ten cycles to failing at cycle 1**. That count is derived from sustained-cycle data, not from a judgement call.

Reading the cases, I found the example was doing *disambiguation* rather than supplying an answer to copy: it resolved element ordering (Mbpp/462), the return container (set vs list, Mbpp/473), rotation direction (Mbpp/586), whether "pairs of tuples" means pairs *of* tuples or the pair *within* one (Mbpp/120, Mbpp/409), and the treatment of zero digits (Mbpp/745). These models **never echo the assert (0.0% of cycles)**, so the leak cannot be a copyable crutch. That refutes my pre-registered H5 and replaces it with a better account.

**A loop the extension broke.** Mbpp/226: *"remove the characters which have odd index values"*, anchored `assert odd_values_string('abcdef') == 'ace'`. Under A0 it survives ten cycles. The summary becomes *"return a string containing only the characters at odd-indexed positions"*, which is coherent but **inverted**. Under A0 the expected output `'ace'` contradicts that reading and the model implements the original anyway; under both ablated arms it fails at cycle 1 with `return string[1::2]` → `'bdf'`. The description was always ambiguous; the oracle was resolving it.

**A loop it did not.** Mbpp/3: *"identify non-prime numbers"*, anchored `assert is_not_prime(2) == False`. Ten cycles under all three anchors; its summary stabilizes at cycle 1 and never moves, because the description fully determines the function. **That is the pattern: the anchor props up precisely the under-specified tasks, and ASL does not distinguish them from the well-specified ones.**

## 5. Limitations and implications

**Threats.** (i) **Test-suite strength.** A loop fails only when the suites catch the drift, so MBPP+'s tests bound this in both directions: weak tests hide real degradation, making sustained-cycle counts an upper bound on fidelity, while the extended EvalPlus inputs can end a loop for reasons unrelated to the anchor. One failure I coded is exactly that, a correct rotation that lost its empty-list guard. Because every arm runs identical suites on identical tasks, this moves all arms together and does not explain the between-arm differences, which is the claim being made. (ii) **Few failure events**, the binding limitation: 4–7 mid-loop failures per baseline arm means each Wilcoxon rests on 11–16 discordant tasks out of 150, which is why a consistent effect reaches significance in one model and not the other. More tasks, not more cycles, is what this design needs. (iii) Three models, two families, and only two ran the intermediate A1 arm, so the A1-vs-A2 decomposition rests on the Gemma pair; the A0-vs-A2 contrast uses all three. (iv) MBPP is from 2021 and these models sit at 0.75–0.77 pass@1, so contamination is plausible, though it would inflate all arms equally. (v) The 150-task subset (mitigated: ρ = 0.937). (vi) A2 changes information content *and* surface form; A1 partly separates them. (vii) Single run per configuration. (viii) One primary coder grading their own experiment; blinding and a sealed second coder mitigate but do not remove this.

**Implications.** EvaLoop's ASL is a sound idea undermined by two design details. The loop converges to a fixed point by cycle 2–3, so ten cycles mostly re-derive the same strings while i² pays compound credit, which is why ASL tracks pass@1 at ρ = 0.945. And the anchor that keeps the harness runnable also re-injects ground truth, which disproportionately rescues under-specified tasks. Both are fixable without abandoning the metric: derive the anchor from the model's own output (A2 costs nothing extra and is strictly more faithful to the self-consistency premise), stop at the first repeated state rather than a fixed budget, and report the sustained-cycle histogram rather than a single i²-weighted scalar. More broadly, a self-consistency benchmark should account for every channel through which the benchmark's own answers can re-enter the loop; here one line of scaffolding, present for a legitimate engineering reason, was quietly supplying the disambiguation that the measurement was supposed to be testing.
