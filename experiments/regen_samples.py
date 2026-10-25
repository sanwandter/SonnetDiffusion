"""Regenerate the paper's main configurations and keep every raw sample.

The original drivers (k_sweep.py, best_of_n.py) stored only aggregate metrics,
so the reviewers' requests -- stricter rhyme, joint satisfaction, an
independent rhyme measure, fluency, a human study -- could not be computed on
the outputs the paper reports. This driver reruns the same configurations
under the same seeds and writes the samples themselves; scoring happens
offline in strict_metrics.py.

Configurations (n sonnets per seed, seeds 102/202/302):
  nocontrol     controller off                         (= k_sweep K=0)
  K5            full stack, K=5                        (= k_sweep K=5, headline)
  K5_repel      full stack, K=5, + group repulsion     (new; lambda_repel fixed
                                                        a priori to lambda_rep)
  pool          n*N_POOL uncontrolled samples, drawn exactly as best_of_n.py
                draws its pool (batches of `chunk`, seed + 1000*sub), so
                best-of-N can be re-selected offline under any metric
  K2            full stack, K=2                        (= k_sweep K=2)
  K5_cdcdcd     full stack, K=5, target ABBA ABBA CDC DCD  (breadth: only the
  K5_cdedce     full stack, K=5, target ABBA ABBA CDE DCE   target changes)

--ckpt-step S samples from ema_0.9999_<S>.pt instead of the paper's 50k
checkpoint and writes under results/samples/ckpt<S/1000>k/ (breadth: more
than one checkpoint).

The order puts the configurations the resubmission needs most first. Each
(config, seed) file is written atomically and skipped on restart, so a reboot
loses at most the unit in flight.

Each unit also records the old-evaluator metrics, so the run doubles as a
reproduction check against k_sweep_results.json: seeding fixes the noise
trajectory but not GPU kernel determinism, so expect close, not identical.

Uso:
  conda run -n gpu python experiments/regen_samples.py               # everything
  conda run -n gpu python experiments/regen_samples.py --only K5 K5_repel
  conda run -n gpu python experiments/regen_samples.py --n 2 --pool-N 2 \\
      --out /tmp/regen_smoke                                          # smoke test
Salida: results/samples/<config>/seed<seed>.txt(.gz) + .json sidecar
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import harness
from rhyme_es import QUARTET_GROUPS

N_POOL = 168   # wall-matched best-of-N in the paper; smaller N nest inside it

CONFIGS = {
    'nocontrol': dict(K=0),
    'K5':        dict(K=5),
    'K5_synal':  dict(K=5, syll_synalepha=True),   # syllable penalty with synalepha
    'K5_repel':  dict(K=5, lambda_repel=harness.FULL['lambda_rep']),
    'pool':      dict(K=0, pool=True),
    'K2':        dict(K=2),
    # Other standard tercet arrangements; quatrains stay ABBA ABBA.
    'K5_cdcdcd': dict(K=5, groups={**QUARTET_GROUPS,
                                   'C': [9, 11, 13], 'D': [10, 12, 14]}),
    'K5_cdedce': dict(K=5, groups={**QUARTET_GROUPS,
                                   'C': [9, 13], 'D': [10, 12], 'E': [11, 14]}),
    'K5_repel_cdcdcd': dict(K=5, lambda_repel=harness.FULL['lambda_rep'],
                            groups={**QUARTET_GROUPS,
                                    'C': [9, 11, 13], 'D': [10, 12, 14]}),
}
# lambda_repel sweep around the a-priori value (K5_repel, 0.2)
for _lam in (0.05, 0.1, 0.4, 0.8):
    CONFIGS[f'K5_repel{_lam:g}'] = dict(K=5, lambda_repel=_lam)


def generate_chunked(ctx, total, seed, langevin_fn, chunk):
    """Same batching and per-batch seeding as best_of_n.generate_chunked."""
    lines, wall, calls, sub = [], 0.0, 0, 0
    while len(lines) < total:
        k = min(chunk, total - len(lines))
        ls, dt, fc = harness.generate(ctx, k, seed + 1000 * sub, langevin_fn)
        lines.extend(ls)
        wall += dt
        calls += fc
        sub += 1
    return lines, wall, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30, help='sonnets per seed')
    ap.add_argument('--pool-N', type=int, default=N_POOL,
                    help='pool size per seed is n * pool-N')
    ap.add_argument('--chunk', type=int, default=30, help='max batch size')
    ap.add_argument('--only', nargs='*', default=None, choices=list(CONFIGS))
    ap.add_argument('--out', default=None,
                    help='default results/samples, or results/samples/ckpt<k>k')
    ap.add_argument('--ckpt-step', type=int, default=None,
                    help='EMA checkpoint step to sample from (default: the paper\'s 50k)')
    a = ap.parse_args()
    ckpt_path = harness.CKPT
    if a.ckpt_step is not None:
        ckpt_path = os.path.join(os.path.dirname(harness.CKPT),
                                 f'ema_0.9999_{a.ckpt_step:06d}.pt')
        if not os.path.exists(ckpt_path):
            sys.exit(f'no checkpoint at {ckpt_path}')
    if a.out is None:
        a.out = os.path.join(ROOT, 'results', 'samples')
        if a.ckpt_step is not None:
            a.out = os.path.join(a.out, f'ckpt{a.ckpt_step // 1000:03d}k')

    names = a.only or list(CONFIGS)
    ctx = None
    for name in names:
        cfg = dict(CONFIGS[name])
        is_pool = cfg.pop('pool', False)
        ext = '.txt.gz' if is_pool else '.txt'
        for seed in harness.SEEDS:
            path = os.path.join(a.out, name, f'seed{seed}{ext}')
            if os.path.exists(path):
                print(f"[{name} seed={seed}] already on disk -- skipping")
                continue
            if ctx is None:
                ctx = harness.load_ckpt(ckpt_path, respacing='ddim100', device='cuda')
            langevin_fn = (None if cfg['K'] == 0
                           else harness.make_full_langevin(ctx, **cfg))
            total = a.n * a.pool_N if is_pool else a.n
            if is_pool:
                lines, dt, fc = generate_chunked(ctx, total, seed, None, a.chunk)
            else:
                lines, dt, fc = harness.generate(ctx, total, seed, langevin_fn)
            m = harness.score_lines(lines)
            meta = dict(config=name, overrides=cfg, seed=seed, n=len(lines),
                        checkpoint=os.path.basename(ckpt_path),
                        wall_s=round(dt, 2), forward_calls=fc,
                        seconds_per_sonnet=round(dt / len(lines), 4),
                        old_metrics={k: round(m[k], 2)
                                     for k in harness.METRIC_KEYS})
            if is_pool:
                # strict_metrics.py re-selects best-of-N from this: groups of
                # N over the first keep*N samples, N <= pool_N
                meta.update(keep=a.n, pool_N=a.pool_N, chunk=a.chunk)
            harness.save_samples(path, lines, meta)
            print(f"[{name} seed={seed}] n={len(lines)} {dt:.0f}s "
                  f"fmt={m['format_pct']:.1f}% rhy={m['rhyme_full_pct']:.1f}% "
                  f"syl={m['syll_in_range_pct']:.1f}% "
                  f"dist={m['distinct_endwords_pct']:.1f}%", flush=True)


if __name__ == '__main__':
    main()
