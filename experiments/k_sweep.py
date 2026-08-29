"""Quality and cost as a function of the inner-step budget K.

K is the number of Langevin inner optimization steps per sampling step. Each
one costs an extra denoiser forward and backward pass, so K is the knob that
trades constraint satisfaction against sampling cost. K=0 is the controller
switched off entirely, which anchors the curve at real measured zero-control
cost instead of an extrapolated point.

Together with best_of_n_results.json this produces the paper's main figure:
rhyme accuracy against wall clock, gradient control versus buying more samples.

Cost is roughly linear in K, so the sweep totals ~7 h at n=30. It checkpoints
after every (K, seed) pair and restarts from where it stopped.

Uso: conda run -n gpu python resultados/experimentos/k_sweep.py [--n 30]
Salida: resultados/datos/k_sweep_results.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import harness

K_VALUES = [0, 1, 2, 3, 5, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', default=None,
                    help='output path; point smoke runs away from the real '
                         'file so a tiny-n sweep cannot be read as the result')
    a = ap.parse_args()

    outp = a.out or os.path.join(ROOT, 'resultados', 'datos', 'k_sweep_results.json')
    partialp = outp.replace('.json', '_partial.json')
    config = dict(n=a.n, seeds=harness.SEEDS, K_values=K_VALUES)
    done = harness.load_partial(partialp, config)
    if done:
        print(f"resuming -- {len(done)}/{len(K_VALUES) * len(harness.SEEDS)} "
              f"(K, seed) pairs already complete")

    ctx = None
    results = []

    for K in K_VALUES:
        per_seed, walls, calls = [], [], []
        for seed in harness.SEEDS:
            key = f"{K}|{seed}"
            cached = done.get(key)
            if cached is not None:
                per_seed.append(cached['metrics'])
                walls.append(cached['wall_s'])
                calls.append(cached['forward_calls'])
                print(f"[K={K} seed={seed}] restored "
                      f"rhy={cached['metrics']['rhyme_full_pct']:.1f}% "
                      f"({cached['wall_s']:.1f}s, {cached['forward_calls']} fwd)")
                continue

            if ctx is None:
                ctx = harness.load_ckpt(respacing='ddim100', device='cuda')
            langevin_fn = None if K == 0 else harness.make_full_langevin(ctx, K=K)
            lines, dt, fc = harness.generate(ctx, a.n, seed, langevin_fn)
            m = harness.score_lines(lines)
            per_seed.append(m)
            walls.append(dt)
            calls.append(fc)
            print(f"[K={K} seed={seed}] fmt={m['format_pct']:.1f}% "
                  f"rhy={m['rhyme_full_pct']:.1f}% ({dt:.1f}s, {fc} fwd)",
                  flush=True)

            # Timing is part of the result here, not just bookkeeping -- this
            # sweep is the x-axis of the cost/quality figure -- so it is
            # checkpointed alongside the metrics.
            done[key] = dict(metrics=m, wall_s=dt, forward_calls=fc)
            harness.save_partial(partialp, config, done)

        results.append(dict(
            K=K,
            per_seed=[{k: round(r[k], 2) for k in harness.METRIC_KEYS}
                      for r in per_seed],
            aggregate=harness.aggregate_seeds(per_seed, harness.METRIC_KEYS),
            wall_s=harness.mean_ci(walls),
            forward_calls=harness.mean_ci([float(c) for c in calls]),
            seconds_per_sonnet=round(sum(walls) / len(walls) / a.n, 2)))

    with open(outp, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    harness.clear_partial(partialp)

    print(f"\n=== BARRIDO DE PASOS INTERNOS K (n={a.n}, 3 semillas) ===")
    print(f"{'K':>3} {'Fmt%':>16} {'RimaFull%':>16} {'s/soneto':>10} {'fwd':>10}")
    for r in results:
        f_ = r['aggregate']['format_pct']
        y_ = r['aggregate']['rhyme_full_pct']
        print(f"{r['K']:>3} {f_['mean']:>8.1f} +/-{f_['halfwidth']:<5.1f} "
              f"{y_['mean']:>8.1f} +/-{y_['halfwidth']:<5.1f} "
              f"{r['seconds_per_sonnet']:>10.2f} "
              f"{r['forward_calls']['mean']:>10.0f}")
    print("Guardado:", outp)


if __name__ == '__main__':
    main()
