# How to code the 20 failures

Budget about 30–45 minutes. Everything you need is in
`CODING_WORKSHEET.md`; the result goes into `failures_to_code.csv`.

## Steps

1. Open **`CODING_WORKSHEET.md`**. It has all 20 failures, each with the
   original task, the summary that fed the failing cycle, the code that passed
   the cycle before, and the code that failed — plus mechanical facts (does it
   parse, does it define the expected function, base vs plus outcome).
2. For each row, decide **one primary code** and write it into the
   `primary_code` column of **`failures_to_code.csv`**, matching `row_id`.
   Use the `notes` column for a one-line reason — those become the examples
   quoted in the report.
3. Run `python scripts/score_coding.py`. It unblinds, derives C6, and reports
   agreement against the sealed second coder.

## The question to ask on each row

The loop runs: original task → code → summary → code → … The **summary** shown
is what the model wrote about its *own passing code*, and it is the only
description the failing cycle received.

- Did the summary **lose a detail** the original stated? → **C1**
- Did the summary describe a **different operation** that is coherent on its
  own? → **C2**
- Was the summary **faithful**, and the code just fails to implement it? → **C3**
- Wrong function **name, arity, or parameter order**? → **C4**
- Not really code — **fence, prose, truncation**, does not parse? → **C5**
- Base tests pass, only the **extended** tests fail? → **C7**
- Timeout, exception, or anything else mechanical? → **C8**

**C6 (oracle dependence) is not yours to assign.** It needs cross-arm
information, so `score_coding.py` derives it from the run data.

## Rules that keep this defensible

- **Do not open `failures_key.json` or `coder_llm_sealed.json` until you are
  done.** The first reveals which arm each failure came from; the second holds
  an independent LLM pass. Either would bias you toward the finding under test.
- One primary code per row. If two apply, pick the one that *caused* the
  failure and put the other in `secondary_code`.
- Disagreeing with the sealed second coder is fine and expected. The
  disagreement is reportable and strengthens the methodology — it is why the
  second pass was sealed rather than deleted.

## When you are done

`score_coding.py` prints the by-arm distribution, the derived C6 count, and
Cohen's kappa. Those numbers replace the `[Primary coding in progress]`
placeholder in `report.md`; rebuild with `python scripts/build_report.py`.
