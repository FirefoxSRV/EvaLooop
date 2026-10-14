"""Report API usage and cost for every run and for the judge.

The assignment requires the model name, prompt, inference settings, how the
output becomes a decision, and the **total and per-example cost** of any LLM used
as a component of the method. Settings and prompts live in the run files and in
the README; this script produces the usage and cost half.

Two numbers are reported, and the difference between them matters:

* **Billed cost** - what was actually spent. Every run here was executed on the
  free tier, so this is 0.00 USD. That is the honest figure, and it is also why
  the experiment was shaped around a request-per-day cap rather than a budget.
* **Cost at paid rates** - what the same token volume would cost if billed. This
  is the figure worth comparing against other setups. It is only printed when
  ``configs/pricing.json`` carries verified rates; absent those, the script says
  so rather than inventing a price.

"Per example" is reported per *task* (the unit a reader thinks in) and per
*cycle* (the unit the loop actually pays for).

Usage:
    python scripts/cost_report.py --results-dir results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pricing(path: Path) -> dict:
    """Load the rate table, tolerating its absence."""
    if not path.exists():
        return {"models": {}, "verified_on": None, "source": None}
    return json.loads(path.read_text())


def cost_of(tokens_in: int, tokens_out: int, rates: dict):
    """Cost at paid rates, or None when the rates are not verified."""
    if not rates:
        return None
    rate_in, rate_out = rates.get("input_per_mtok"), rates.get("output_per_mtok")
    if rate_in is None or rate_out is None:
        return None
    return tokens_in / 1e6 * rate_in + tokens_out / 1e6 * rate_out


# Arms run in this order within a model, baseline first, so the later arms can
# replay cycle 1 from cache.
ARM_ORDER = {"assert": 0, "masked_assert": 1, "signature": 2}

# Counters that accumulate on the shared client across a model's arms.
CUMULATIVE_FIELDS = ("api_calls", "cache_hits", "prompt_tokens", "output_tokens",
                     "thoughts_tokens", "retries", "rate_limited")


def decumulate(rows):
    """Turn a model's cumulative usage counters into per-arm figures.

    The three arms of a model share one client, so its usage counters keep
    running across them: the signature arm's "calls" includes the two arms
    before it. Reporting those as per-arm costs would overstate every arm but
    the first and triple-count the model's total. Differencing them in run order
    recovers the per-arm figures, and the last arm's cumulative value is the
    model's true total.

    A counter that is not monotonically non-decreasing means the arms did not
    share a client (a resumed run, say), so that model is left untouched.
    """
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    out = []
    for model_rows in by_model.values():
        ordered = sorted(model_rows, key=lambda r: ARM_ORDER.get(r["configuration"], 9))
        monotonic = all(
            ordered[i][field] <= ordered[i + 1][field]
            for i in range(len(ordered) - 1) for field in CUMULATIVE_FIELDS
        )
        if not monotonic or len(ordered) < 2:
            out.extend(ordered)
            continue
        previous = {field: 0 for field in CUMULATIVE_FIELDS}
        for row in ordered:
            adjusted = dict(row)
            for field in CUMULATIVE_FIELDS:
                adjusted[field] = row[field] - previous[field]
                previous[field] = row[field]
            adjusted["counters_decumulated"] = True
            adjusted["paid_equivalent_usd"] = cost_of(
                adjusted["prompt_tokens"], adjusted["output_tokens"],
                {} if row["paid_equivalent_usd"] is None else
                {"input_per_mtok": None, "output_per_mtok": None},
            )
            out.append(adjusted)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pricing", default="configs/pricing.json")
    parser.add_argument("--out", default="results/cost.json")
    args = parser.parse_args()

    pricing = load_pricing(Path(args.pricing))
    rates_by_model = pricing.get("models", {})

    rows, judges = [], []
    for path in sorted(Path(args.results_dir).glob("*__*.json")):
        payload = json.loads(path.read_text())
        if "tasks" not in payload or "configuration" not in payload:
            continue
        meta = payload.get("run_meta", {})
        usage = meta.get("usage", {}) or {}
        model = payload["model_id"]
        n_tasks = payload["n_tasks"]
        cycles_run = sum(len(t["cycles"]) for t in payload["tasks"])

        rows.append({
            "model": model,
            "configuration": payload["configuration"],
            "n_tasks": n_tasks,
            "cycles_run": cycles_run,
            # Calls actually made to the provider; cache hits cost nothing, which
            # is why the generation arms after the first are so much cheaper.
            "api_calls": usage.get("calls", 0),
            "cache_hits": (usage.get("cache") or {}).get("hits", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "thoughts_tokens": usage.get("thoughts_tokens", 0),
            "retries": usage.get("retries", 0),
            "rate_limited": usage.get("rate_limited", 0),
            "free_tier": bool(rates_by_model.get(model, {}).get("free_tier", True)),
            "paid_equivalent_usd": cost_of(usage.get("prompt_tokens", 0),
                                           usage.get("output_tokens", 0),
                                           rates_by_model.get(model, {})),
            "wall_seconds": meta.get("wall_seconds"),
        })

        judge = meta.get("judge")
        if judge:
            j_usage = judge.get("usage", {}) or {}
            judges.append({
                "run": path.stem,
                "judge_model": judge.get("model"),
                "published_judge": judge.get("published_judge"),
                "comparable_to_published": judge.get("comparable_to_published"),
                "temperature": judge.get("temperature"),
                "max_tokens": judge.get("max_tokens"),
                "calls": j_usage.get("calls", 0),
                "prompt_tokens": j_usage.get("prompt_tokens", 0),
                "output_tokens": j_usage.get("output_tokens", 0),
                "paid_equivalent_usd": cost_of(j_usage.get("prompt_tokens", 0),
                                               j_usage.get("output_tokens", 0),
                                               rates_by_model.get(judge.get("model"), {})),
            })

    if not rows:
        print(f"No run files in {args.results_dir}", file=sys.stderr)
        return 2

    rows = decumulate(rows)

    print("\nPer-arm usage (counters de-cumulated: the arms of a model share one client)")
    print(f"\n{'model':<24}{'arm':<15}{'calls':>7}{'hits':>7}{'in tok':>9}"
          f"{'out tok':>9}{'tok/task':>9}{'tok/cycle':>10}")
    for row in rows:
        total_tokens = row["prompt_tokens"] + row["output_tokens"]
        print(f"{row['model']:<24}{row['configuration']:<15}{row['api_calls']:>7}"
              f"{row['cache_hits']:>7}{row['prompt_tokens']:>9}{row['output_tokens']:>9}"
              f"{total_tokens / max(row['n_tasks'], 1):>9.0f}"
              f"{total_tokens / max(row['cycles_run'], 1):>10.0f}")

    totals = {
        "api_calls": sum(r["api_calls"] for r in rows),
        "cache_hits": sum(r["cache_hits"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "retries": sum(r["retries"] for r in rows),
        "rate_limited": sum(r["rate_limited"] for r in rows),
        "n_task_runs": sum(r["n_tasks"] for r in rows),
    }
    for j in judges:
        totals["api_calls"] += j["calls"]
        totals["prompt_tokens"] += j["prompt_tokens"]
        totals["output_tokens"] += j["output_tokens"]

    all_tokens = totals["prompt_tokens"] + totals["output_tokens"]
    print(f"\nTotals across {len(rows)} run(s)" + (f" plus {len(judges)} judge pass(es)" if judges else ""))
    print(f"  API calls made        : {totals['api_calls']}")
    print(f"  served from cache     : {totals['cache_hits']}")
    print(f"  tokens in / out       : {totals['prompt_tokens']} / {totals['output_tokens']}")
    print(f"  retries (throttling)  : {totals['retries']} "
          f"({totals['rate_limited']} rate-limited)")

    paid = [r["paid_equivalent_usd"] for r in rows] + [j["paid_equivalent_usd"] for j in judges]
    print(f"\n  Billed cost           : 0.00 USD (all runs on the free tier)")
    if all(p is None for p in paid):
        print(f"  Cost at paid rates    : not computed - no verified rates in "
              f"{args.pricing}")
        print(f"                          (per-task token cost is reported above, so a "
              f"rate can be applied later)")
    else:
        total_paid = sum(p for p in paid if p is not None)
        print(f"  Cost at paid rates    : {total_paid:.4f} USD "
              f"(rates verified {pricing.get('verified_on')})")
        print(f"  per task-run          : {total_paid / max(totals['n_task_runs'], 1):.6f} USD")

    if judges:
        print(f"\nJudge usage (reported per the assignment's LLM-usage rule):")
        for j in judges:
            print(f"  {j['run']}: {j['judge_model']} temp={j['temperature']} "
                  f"max_tokens={j['max_tokens']} calls={j['calls']} "
                  f"comparable_to_published={j['comparable_to_published']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "billed_usd": 0.0,
        "billed_note": "All runs executed on the Gemini API free tier.",
        "pricing_verified_on": pricing.get("verified_on"),
        "pricing_source": pricing.get("source"),
        "totals": totals, "runs": rows, "judge_passes": judges,
    }, indent=2) + "\n")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
