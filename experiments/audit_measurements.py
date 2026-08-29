"""Measurement audit: real parameter count, real controlled-vs-uncontrolled
cost, and the resulting compute-matched best-of-N budget.

Resolves two stated-but-unverified numbers in the course paper: the "87M
parameters" figure (inherited from PoetryDiffusion, never measured here) and
the sampling time, which the paper reports as ~270 s where
resultados/README.md reports ~140 s for the same configuration.

Uso: conda run -n gpu python resultados/experimentos/audit_measurements.py [--n 8] [--repeats 3]
Salida: resultados/datos/audit_measurements.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()

    ctx = harness.load_ckpt(respacing='ddim100', device='cuda')
    model = ctx['model']

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total     : {params_total:,} ({params_total / 1e6:.2f}M)")
    print(f"params trainable : {params_trainable:,} ({params_trainable / 1e6:.2f}M)")

    langevin_fn = harness.make_full_langevin(ctx)

    runs = {'uncontrolled': [], 'controlled': []}
    for r in range(a.repeats):
        seed = harness.SEEDS[r % len(harness.SEEDS)]
        _, dt_u, fc_u = harness.generate(ctx, a.n, seed, None)
        runs['uncontrolled'].append(dict(wall_s=dt_u, forward_calls=fc_u))
        print(f"[rep {r}] uncontrolled {dt_u:.1f}s  {fc_u} fwd", flush=True)

        _, dt_c, fc_c = harness.generate(ctx, a.n, seed, langevin_fn)
        runs['controlled'].append(dict(wall_s=dt_c, forward_calls=fc_c))
        print(f"[rep {r}] controlled   {dt_c:.1f}s  {fc_c} fwd", flush=True)

    wu = harness.mean_ci([r['wall_s'] for r in runs['uncontrolled']])
    wc = harness.mean_ci([r['wall_s'] for r in runs['controlled']])
    fu = harness.mean_ci([float(r['forward_calls']) for r in runs['uncontrolled']])
    fc = harness.mean_ci([float(r['forward_calls']) for r in runs['controlled']])

    ratio_wall = wc['mean'] / wu['mean']
    ratio_fwd = fc['mean'] / fu['mean']
    budget = max(2, int(round(ratio_wall)))

    out = dict(
        n=a.n, repeats=a.repeats,
        params_total=params_total, params_trainable=params_trainable,
        params_millions=round(params_total / 1e6, 2),
        uncontrolled=dict(wall_s=wu, forward_calls=fu, raw=runs['uncontrolled']),
        controlled=dict(wall_s=wc, forward_calls=fc, raw=runs['controlled']),
        ratio_walltime=round(ratio_wall, 3),
        ratio_forward_calls=round(ratio_fwd, 3),
        best_of_n_budget=budget,
        seconds_per_sonnet_controlled=round(wc['mean'] / a.n, 2),
        seconds_per_sonnet_uncontrolled=round(wu['mean'] / a.n, 2))

    outp = os.path.join(ROOT, 'resultados', 'datos', 'audit_measurements.json')
    with open(outp, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n=== AUDIT ===")
    print(f"params              : {out['params_millions']}M")
    print(f"uncontrolled        : {wu['mean']:.1f}s +/- {wu['halfwidth']:.1f}  "
          f"({out['seconds_per_sonnet_uncontrolled']}s/soneto)")
    print(f"controlled          : {wc['mean']:.1f}s +/- {wc['halfwidth']:.1f}  "
          f"({out['seconds_per_sonnet_controlled']}s/soneto)")
    print(f"ratio wall-clock    : {ratio_wall:.2f}x")
    print(f"ratio forward calls : {ratio_fwd:.2f}x")
    print(f"best-of-N budget    : N = {budget}")
    print("Guardado:", outp)


if __name__ == '__main__':
    main()
