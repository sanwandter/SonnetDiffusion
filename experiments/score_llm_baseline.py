"""Score a prompted-model baseline with the same metrics used for every
other system in the paper.

Parameterised on purpose: the Dream-7B baseline (Task 8) reuses this script
by passing different arguments rather than copying it, so there is exactly
one scoring path for all prompted baselines.

Two ways to get the spread across replicates, and they are NOT equivalent:

  independent-runs (preferred) -- pass several --src files, each one a
    separate generation in a fresh context. Variation across them is model
    sampling variation, which is what a confidence interval should describe.

  ordered-blocks (fallback) -- pass one file; it is cut into 3 consecutive
    blocks. This is much weaker, and on the first Haiku run it was actively
    misleading: blocks scored 60.7 / 61.3 / 22.0 on rhyme, because one context
    generated all 30 sonnets and degraded over the last third. That is
    generation-order decay being reported as sampling variance. The mode is
    recorded in the output so the paper cannot quote one as the other.

Uso:
  conda run -n gpu python experiments/score_llm_baseline.py \
      --src results/llm_baseline/haiku45_run1.txt \
            results/llm_baseline/haiku45_run2.txt \
            results/llm_baseline/haiku45_run3.txt
  conda run -n gpu python experiments/score_llm_baseline.py \
      --src results/dream_baseline/dream7b_sonnets.txt \
      --out results/dream_baseline_metrics.json \
      --model Dream-org/Dream-v0-Instruct-7B
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import harness

# The reported Haiku baseline is the single 30-sonnet run scored as ordered
# blocks. The three-independent-run variant exists on disk and is the stronger
# protocol, but it is deliberately NOT the default: the reported baseline is
# the n=30 one, and a default that regenerates a different set of numbers
# would silently contradict the paper. Pass the three files explicitly to use
# them.
DEFAULT_SRC = [os.path.join(ROOT, 'results', 'llm_baseline',
                            'haiku45_sonnets.txt')]
DEFAULT_OUT = os.path.join(ROOT, 'results',
                           'llm_baseline_metrics.json')
DEFAULT_MODEL = 'claude-haiku-4-5-20251001'


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='+', default=DEFAULT_SRC)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    a = ap.parse_args()

    per_file = [read_lines(p) for p in a.src]
    for p, ls in zip(a.src, per_file):
        print(f"loaded {len(ls)} sonnets from {p}")
    all_lines = [l for ls in per_file for l in ls]

    if len(a.src) > 1:
        kind = 'independent-runs'
        replicates = per_file
    else:
        kind = 'ordered-blocks'
        lines = per_file[0]
        if len(lines) < 3:
            raise ValueError(f"need at least 3 sonnets to form blocks, got {len(lines)}")
        third = len(lines) // 3
        replicates = [lines[0:third], lines[third:2 * third], lines[2 * third:]]
        print("WARNING: single source file -- falling back to ordered blocks. "
              "The resulting interval mixes generation-order decay into the "
              "spread and understates how good the model actually is.")

    overall = harness.score_lines(all_lines)
    per_rep = [harness.score_lines(r) for r in replicates]
    agg = harness.aggregate_seeds(per_rep, harness.METRIC_KEYS)

    rounded = [{k: round(r[k], 2) for k in harness.METRIC_KEYS} for r in per_rep]
    out = dict(model=a.model, n=len(all_lines), prompted=True, finetuned=False,
               sources=list(a.src), replicate_kind=kind,
               metrics={k: round(overall[k], 2) for k in harness.METRIC_KEYS},
               per_replicate=rounded,
               aggregate=agg)
    if kind == 'ordered-blocks':
        out['per_block'] = rounded      # keep the old key readable
    with open(a.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n=== PROMPTED BASELINE ({a.model}, n={len(all_lines)}, {kind}) ===")
    for k in harness.METRIC_KEYS:
        c = agg[k]
        print(f"{k:<24} {overall[k]:8.2f}   "
              f"({len(per_rep)} reps {c['mean']:.2f} +/- {c['halfwidth']:.2f})")
    print("Guardado:", a.out)


if __name__ == '__main__':
    main()
