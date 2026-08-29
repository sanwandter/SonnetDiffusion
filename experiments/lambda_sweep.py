"""Sweep a single controller weight and report the whole curve.

Motivated by the ablation: of the six penalties, only rhyme and syllable move
any metric, and the rhyme penalty is what causes the end-word collapse (the
arm without it scores 99.2 on distinct end words against the full stack's
83.0). Two questions follow, and each is one dimension, not a grid:

  lambda_rhyme -- what does the rhyme/diversity trade-off actually look like?
    A grid over all six weights would spend most of its GPU confirming that
    the four inert penalties are inert.

  lambda_rep -- the repetition penalty fails at the one job it has. Is it
    merely underweighted at 0.2, or is the (p_i . p_j)^2 formulation wrong?
    If raising it tenfold still does not restore end words, the formulation
    is at fault, which is the more interesting result.

The whole curve gets reported, never the argmax. These weights were inherited
from the course project rather than tuned, which is what makes them honest;
picking the best value on the metric the paper reports and presenting it as
the headline would be tuning on the test set.

The point where the swept value equals the default is the ablation's `full`
arm -- same weights, same n, same seeds -- so it is read from
ablation_seeded.json instead of being generated a second time.

Uso:
  conda run -n gpu python resultados/experimentos/lambda_sweep.py \
      --param lambda_rhyme --values 0.1 0.25 0.5 1.0 2.0
  conda run -n gpu python resultados/experimentos/lambda_sweep.py \
      --param lambda_rep --values 0.2 1.0 5.0
Salida: resultados/datos/sweep_<param>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import harness


def load_full_arm():
    """The ablation's `full` arm, or None. Same config as the default point."""
    path = os.path.join(ROOT, 'resultados', 'datos', 'ablation_seeded.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for entry in json.load(f):
            if entry['config'] == 'full':
                return entry
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', required=True, choices=sorted(harness.FULL),
                    help='controller weight to sweep')
    ap.add_argument('--values', required=True, type=float, nargs='+')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    default = harness.FULL[a.param]
    outp = a.out or os.path.join(ROOT, 'resultados', 'datos',
                                 f'sweep_{a.param}.json')
    partialp = outp.replace('.json', '_partial.json')
    config = dict(param=a.param, values=a.values, n=a.n, seeds=harness.SEEDS)
    done = harness.load_partial(partialp, config)
    if done:
        print(f"resuming -- {len(done)}/{len(a.values) * len(harness.SEEDS)} "
              f"(value, seed) pairs already complete")

    full_arm = load_full_arm()
    print(f"sweeping {a.param} over {a.values} (default {default})")
    if full_arm is None:
        print("  ablation_seeded.json not found -- the default point will be "
              "generated rather than reused")

    ctx = None
    results = []

    for value in a.values:
        per_seed = []

        # The default point is the ablation's full arm. Reusing it is not a
        # shortcut: it is the same weights, n and seeds, so regenerating would
        # cost ~35 min per seed to reproduce numbers already on disk.
        if value == default and full_arm is not None:
            per_seed = list(full_arm['per_seed'])
            print(f"[{a.param}={value}] reused from ablation `full` "
                  f"(rhyme {full_arm['aggregate']['rhyme_full_pct']['mean']:.1f})")
            results.append(dict(param=a.param, value=value, source='ablation-full',
                                per_seed=per_seed, aggregate=full_arm['aggregate']))
            continue

        for seed in harness.SEEDS:
            key = f"{value}|{seed}"
            cached = done.get(key)
            if cached is not None:
                per_seed.append(cached)
                print(f"[{a.param}={value} seed={seed}] restored "
                      f"rhy={cached['rhyme_full_pct']:.1f}%")
                continue

            if ctx is None:
                ctx = harness.load_ckpt(respacing='ddim100', device='cuda')
            langevin_fn = harness.make_full_langevin(ctx, **{a.param: value})
            lines, dt, fc = harness.generate(ctx, a.n, seed, langevin_fn)
            m = harness.score_lines(lines)
            m['wall_s'] = round(dt, 1)
            per_seed.append(m)
            print(f"[{a.param}={value} seed={seed}] fmt={m['format_pct']:.1f}% "
                  f"rhy={m['rhyme_full_pct']:.1f}% "
                  f"syll={m['syll_in_range_pct']:.1f}% "
                  f"endw={m['distinct_endwords_pct']:.1f}% ({dt:.1f}s)", flush=True)
            done[key] = m
            harness.save_partial(partialp, config, done)

        results.append(dict(
            param=a.param, value=value, source='generated',
            per_seed=[{k: round(r[k], 2) for k in harness.METRIC_KEYS}
                      for r in per_seed],
            aggregate=harness.aggregate_seeds(per_seed, harness.METRIC_KEYS)))

    with open(outp, 'w') as f:
        json.dump(dict(param=a.param, default=default, n=a.n,
                       seeds=harness.SEEDS, points=results),
                  f, ensure_ascii=False, indent=1)
    harness.clear_partial(partialp)

    print(f"\n=== BARRIDO DE {a.param} (n={a.n}, 3 semillas) ===")
    print(f"{'valor':>8} {'Rima%':>16} {'Silabas%':>16} {'FinalesDist%':>16}")
    for r in results:
        g = r['aggregate']
        mark = ' *' if r['value'] == default else ''
        print(f"{r['value']:>8} "
              f"{g['rhyme_full_pct']['mean']:>8.1f} +/-{g['rhyme_full_pct']['halfwidth']:<5.1f} "
              f"{g['syll_in_range_pct']['mean']:>8.1f} +/-{g['syll_in_range_pct']['halfwidth']:<5.1f} "
              f"{g['distinct_endwords_pct']['mean']:>8.1f} +/-{g['distinct_endwords_pct']['halfwidth']:<5.1f}"
              f"{mark}")
    print("  * valor por defecto (reusado de la ablacion)")
    print("Guardado:", outp)


if __name__ == '__main__':
    main()
