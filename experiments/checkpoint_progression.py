"""Quality vs training step, over the checkpoints that still exist on disk.

The course paper reports steps 55k-70k; those checkpoint files are not in the
repository and the rows cannot be reproduced. This runs the EMA checkpoints
that do exist (35k-50k) under the full controller stack, seeded.

Checkpoints after every (step, seed) pair -- 12 pairs at ~35 min each is ~7 h,
too long to risk losing to a reboot.

Uso: conda run -n gpu python resultados/experimentos/checkpoint_progression.py [--n 30]
Salida: resultados/datos/checkpoint_progression.json
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import harness

CKPT_DIR = os.path.join(ROOT, 'improved-diffusion', 'diffusion_models',
                        'diff_e2e-tgt_pad_2000steps_16dim_sonnet_es')


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', default=None,
                    help='output path; point smoke runs away from the real '
                         'file so a tiny-n sweep cannot be read as the result')
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(CKPT_DIR, 'ema_0.9999_*.pt')))
    print("checkpoints found:", [os.path.basename(p) for p in paths])

    outp = a.out or os.path.join(ROOT, 'resultados', 'datos',
                                 'checkpoint_progression.json')
    partialp = outp.replace('.json', '_partial.json')
    config = dict(n=a.n, seeds=harness.SEEDS,
                  checkpoints=[os.path.basename(p) for p in paths])
    done = harness.load_partial(partialp, config)
    if done:
        print(f"resuming -- {len(done)}/{len(paths) * len(harness.SEEDS)} "
              f"(step, seed) pairs already complete")

    results = []
    for path in paths:
        step = int(re.search(r'(\d+)\.pt$', path).group(1))
        pending = [s for s in harness.SEEDS if f"{step}|{s}" not in done]

        # Loading a checkpoint costs GPU memory and time; skip it entirely when
        # every seed for this step is already on disk.
        ctx = None
        if pending:
            ctx = harness.load_ckpt(ckpt_path=path)
            langevin_fn = harness.make_full_langevin(ctx)

        per_seed = []
        for seed in harness.SEEDS:
            key = f"{step}|{seed}"
            cached = done.get(key)
            if cached is not None:
                per_seed.append(cached)
                print(f"[{step} seed={seed}] restored "
                      f"fmt={cached['format_pct']:.1f}% rhy={cached['rhyme_full_pct']:.1f}%")
                continue
            lines, dt, fc = harness.generate(ctx, a.n, seed, langevin_fn)
            m = harness.score_lines(lines)
            per_seed.append(m)
            print(f"[{step} seed={seed}] fmt={m['format_pct']:.1f}% "
                  f"rhy={m['rhyme_full_pct']:.1f}%", flush=True)
            done[key] = m
            harness.save_partial(partialp, config, done)

        results.append(dict(
            step=step, kind='ema',
            per_seed=[{k: round(r[k], 2) for k in harness.METRIC_KEYS} for r in per_seed],
            aggregate=harness.aggregate_seeds(per_seed, harness.METRIC_KEYS)))
        if ctx is not None:
            del ctx
            torch.cuda.empty_cache()

    with open(outp, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    harness.clear_partial(partialp)

    print(f"\n=== PROGRESION POR CHECKPOINT (EMA, n={a.n}, 3 semillas) ===")
    print(f"{'Paso':>7} {'Fmt%':>16} {'RimaFull%':>16}")
    for r in results:
        f_ = r['aggregate']['format_pct']
        y_ = r['aggregate']['rhyme_full_pct']
        print(f"{r['step']:>7} {f_['mean']:>8.1f} +/-{f_['halfwidth']:<5.1f} "
              f"{y_['mean']:>8.1f} +/-{y_['halfwidth']:<5.1f}")
    print("Guardado:", outp)


if __name__ == '__main__':
    main()
