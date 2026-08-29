"""Seeded controller ablation with confidence intervals.

Same 8 arms as ablation_controllers.py, but run across 3 seeds at n=30 so the
paper can report error bars. The original n=6 unseeded results stay in
ablation_results.json for provenance; this writes a separate file.

Cost warning: dropping one lambda does not make a run cheaper -- the controller
still takes K=5 inner steps on every respaced step -- so each of the 7
controlled arms costs a full controlled generation, measured at ~35 min for
n=30. The whole sweep is therefore ~12 h, not the "several hours" first
estimated. It checkpoints after every (arm, seed) pair, so it can be killed and
restarted freely; only the pair in flight is lost.

Uso: conda run -n gpu python resultados/experimentos/ablation_seeded.py [--n 30]
Salida: resultados/datos/ablation_seeded.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import harness

ABLATIONS = [
    ('full',        {}),
    ('-formato',    {'lambda_fmt': 0.0}),
    ('-rima',       {'lambda_rhyme': 0.0}),
    ('-repeticion', {'lambda_rep': 0.0}),
    ('-silaba',     {'lambda_syll': 0.0}),
    ('-concepto',   {'lambda_concept': 0.0}),
    ('-emocion',    {'lambda_emotion': 0.0}),
    ('sin-control', None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', default=None,
                    help='output path; smoke runs should point this away from '
                         'the real file so a partial sweep at tiny n cannot be '
                         'picked up by build_tables.py as if it were the result')
    a = ap.parse_args()

    outp = a.out or os.path.join(ROOT, 'resultados', 'datos', 'ablation_seeded.json')
    partialp = outp.replace('.json', '_partial.json')
    config = dict(n=a.n, arms=[name for name, _ in ABLATIONS], seeds=harness.SEEDS)
    done = harness.load_partial(partialp, config)
    if done:
        print(f"resuming -- {len(done)}/{len(ABLATIONS) * len(harness.SEEDS)} "
              f"(arm, seed) pairs already complete")

    # The model is only needed if something is left to generate; a fully
    # resumed run should not pay to load it just to reprint a table.
    ctx = None
    results = []

    for name, override in ABLATIONS:
        per_seed = []
        for seed in harness.SEEDS:
            key = f"{name}|{seed}"
            cached = done.get(key)
            if cached is not None:
                per_seed.append(cached)
                print(f"[{name} seed={seed}] restored "
                      f"fmt={cached['format_pct']:.1f}% rhy={cached['rhyme_full_pct']:.1f}%")
                continue

            if ctx is None:
                ctx = harness.load_ckpt(respacing='ddim100', device='cuda')
            langevin_fn = (None if override is None
                           else harness.make_full_langevin(ctx, **override))
            lines, dt, fc = harness.generate(ctx, a.n, seed, langevin_fn)
            m = harness.score_lines(lines)
            m['wall_s'] = round(dt, 1)
            m['forward_calls'] = fc
            per_seed.append(m)
            print(f"[{name} seed={seed}] fmt={m['format_pct']:.1f}% "
                  f"rhy={m['rhyme_full_pct']:.1f}% ({dt:.1f}s)", flush=True)

            done[key] = m
            harness.save_partial(partialp, config, done)

        results.append(dict(
            config=name,
            per_seed=[{k: round(r[k], 2) for k in harness.METRIC_KEYS} for r in per_seed],
            aggregate=harness.aggregate_seeds(per_seed, harness.METRIC_KEYS),
            wall_s=[r['wall_s'] for r in per_seed]))

    with open(outp, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    harness.clear_partial(partialp)

    print(f"\n=== ABLACION SEMILLADA (50k EMA, n={a.n}, 3 semillas) ===")
    print(f"{'Config':<13} {'Fmt%':>16} {'RimaFull%':>16}")
    for r in results:
        f_ = r['aggregate']['format_pct']
        y_ = r['aggregate']['rhyme_full_pct']
        print(f"{r['config']:<13} {f_['mean']:>8.1f} +/-{f_['halfwidth']:<5.1f} "
              f"{y_['mean']:>8.1f} +/-{y_['halfwidth']:<5.1f}")
    print("Guardado:", outp)


if __name__ == '__main__':
    main()
