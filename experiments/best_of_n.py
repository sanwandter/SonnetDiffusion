"""Compute-matched oracle best-of-N.

The question a reviewer will ask first: does gradient control beat simply
sampling more from the same model? To answer it as strongly as possible the
baseline is made deliberately hard:

  * selection is ORACLE -- each group of N candidates is scored with the exact
    rhyme metric the paper reports, and the argmax is kept. Beating a baseline
    that gets to cheat on the metric is a much stronger claim than beating a
    random draw.
  * the budget is matched TWO ways, because the audit showed the two natural
    definitions differ by more than an order of magnitude:

      N_fwd  = 6   denoiser forward passes  (550 vs 100 -> 5.5x)
      N_wall = 69  wall-clock seconds       (263.9s vs 3.85s -> 68.5x)

    Reporting only N_fwd would flatter the method; reporting only N_wall
    would damn it. The gap is an implementation artifact -- the controlled
    path builds an autograd graph, runs K=5 sequential inner steps that
    cannot be batched, and materialises a dense [B, 256, 23001] logit tensor
    on each one -- not a property of gradient guidance itself. Forward-pass
    count is the hardware- and implementation-independent measure; wall clock
    is what a practitioner actually pays today. Both go in the paper.

Both arms are drawn from ONE pool of n*N_wall uncontrolled samples: the
N_fwd arm oracle-selects over disjoint groups of 6 taken from the first
n*N_fwd elements, the N_wall arm over disjoint groups of 69 across the whole
pool. The samples are i.i.d., so reusing the pool is sound and saves an
entire generation pass.

Uso: conda run -n gpu python experiments/best_of_n.py [--n 30] [--chunk 30]
Salida: results/best_of_n_results.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import harness


def generate_chunked(ctx, total, seed, langevin_fn, chunk):
    """Generate `total` samples in batches of at most `chunk` (24GB limit)."""
    lines, wall, calls = [], 0.0, 0
    produced = 0
    sub = 0
    while produced < total:
        k = min(chunk, total - produced)
        ls, dt, fc = harness.generate(ctx, k, seed + 1000 * sub, langevin_fn)
        lines.extend(ls)
        wall += dt
        calls += fc
        produced += k
        sub += 1
    return lines, wall, calls


def load_partial(path, n, n_values):
    """Resume state from a previous run, if it matches this configuration.

    A full sweep is ~50 min of GPU per seed, so losing the whole run to a
    reboot is expensive. State is checkpointed after every seed; a rerun with
    the same --n and the same N sweep picks up where it stopped. A mismatch is
    treated as a fresh start rather than an error, so changing --n does not
    require remembering to delete the file.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        p = json.load(f)
    if p.get('n_per_seed') != n or p.get('N_values') != n_values:
        print(f"partial state is for a different config -- ignoring {path}")
        return {}
    return p.get('seeds_done', {})


def save_partial(path, n, n_values, seeds_done):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dict(n_per_seed=n, N_values=n_values, seeds_done=seeds_done),
                  f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)   # atomic -- a crash mid-write cannot corrupt state


def oracle_select(scored, group_size, keep):
    """Oracle best-of-N selection over a pre-scored pool.

    `scored` is a list of (line, rhyme, is_fmt) tuples -- pre-scored once so a
    sweep over several group sizes does not rescore the same pool repeatedly.
    Takes the first `keep * group_size` entries, splits them into consecutive
    groups, and keeps the highest-rhyme sonnet from each. Ties break toward
    exact-format sonnets, then toward the first.
    """
    chosen = []
    needed = keep * group_size
    for i in range(0, needed, group_size):
        group = scored[i:i + group_size]
        if not group:
            continue
        best = max(group, key=lambda t: (t[1], t[2]))
        chosen.append(best[0])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30, help='sonnets kept per seed')
    ap.add_argument('--chunk', type=int, default=30, help='max batch size')
    a = ap.parse_args()

    auditp = os.path.join(ROOT, 'results', 'audit_measurements.json')
    with open(auditp) as f:
        audit = json.load(f)

    # Forward-call matching is batch-size independent: 550 vs 100 per batch is
    # 5.5x however many sonnets are in flight. Wall clock is NOT. The audit
    # timed batches of 8 and got 32.98 s/sonnet controlled, but every
    # experiment here runs batches of 30, where it is 70.58 -- the controlled
    # path holds an autograd graph over K=5 inner steps and materialises a
    # [B, 256, 23001] logit tensor, so the larger batch is more than twice as
    # expensive per sonnet. Matching on the audit's ratio gave the baseline 41%
    # of the compute it was supposed to match, in our favour.
    #
    # So the wall-clock budget comes from k_sweep_results.json when it exists,
    # which measured both arms at the batch size actually used.
    n_fwd = int(round(audit['ratio_forward_calls']))
    n_wall = audit['best_of_n_budget']
    budget_src = 'audit_measurements.json (batch 8)'
    kpath = os.path.join(ROOT, 'results', 'k_sweep_results.json')
    if os.path.exists(kpath):
        with open(kpath) as f:
            ks = {r['K']: r for r in json.load(f)}
        if 0 in ks and 5 in ks:
            ctrl_s = ks[5]['seconds_per_sonnet']
            unctrl_s = ks[0]['seconds_per_sonnet']
            n_wall = int(round(ctrl_s / unctrl_s))
            budget_src = (f'k_sweep_results.json (batch {a.n}): '
                          f'{ctrl_s} / {unctrl_s} s per sonnet')
    print(f"wall-clock budget from {budget_src}")

    N_VALUES = sorted({1, 2, n_fwd, (n_fwd + n_wall) // 4, n_wall})
    n_max = max(N_VALUES)
    print(f"forward-call-matched N={n_fwd}, wall-clock-matched N={n_wall}")
    print(f"sweeping N over {N_VALUES} from one pool of {a.n * n_max} per seed")

    partialp = os.path.join(ROOT, 'results', 'best_of_n_partial.json')
    seeds_done = load_partial(partialp, a.n, N_VALUES)
    if seeds_done:
        print(f"resuming -- seeds already complete: "
              f"{sorted(int(s) for s in seeds_done)}")

    ctx = None
    if len(seeds_done) < len(harness.SEEDS):
        ctx = harness.load_ckpt(respacing='ddim100', device='cuda')
        langevin_fn = harness.make_full_langevin(ctx)

    ctrl_metrics, ctrl_timing = [], []
    bon = {N: [] for N in N_VALUES}
    pool_timing = []

    for seed in harness.SEEDS:
        cached = seeds_done.get(str(seed))
        if cached is not None:
            ctrl_metrics.append(cached['controlled'])
            ctrl_timing.append(cached['ctrl_timing'])
            pool_timing.append(cached['pool_timing'])
            for N in N_VALUES:
                bon[N].append(cached['best_of_n'][str(N)])
            print(f"[seed {seed}] restored from partial state "
                  f"(controlled rhy={cached['controlled']['rhyme_full_pct']:.1f}%)")
            continue

        ls, dt, fc = generate_chunked(ctx, a.n, seed, langevin_fn, a.chunk)
        ctrl_metrics.append(harness.score_lines(ls))
        ctrl_timing.append(dict(wall_s=round(dt, 1), forward_calls=fc))
        print(f"[seed {seed}] controlled {dt:.1f}s "
              f"rhy={ctrl_metrics[-1]['rhyme_full_pct']:.1f}%", flush=True)

        pool, dt, fc = generate_chunked(ctx, a.n * n_max, seed, None, a.chunk)
        pool_timing.append(dict(wall_s=round(dt, 1), forward_calls=fc,
                                pool_size=len(pool)))
        print(f"[seed {seed}] pool of {len(pool)} in {dt:.1f}s -- scoring",
              flush=True)

        scored = []
        for line in pool:
            s = harness.score_one(line)
            scored.append((line, s['rhyme'], s['n_lines'] == 14))

        for N in N_VALUES:
            picked = oracle_select(scored, N, a.n)
            bon[N].append(harness.score_lines(picked))
            print(f"    best-of-{N:<3d} rhy={bon[N][-1]['rhyme_full_pct']:.1f}%",
                  flush=True)

        seeds_done[str(seed)] = dict(
            controlled=ctrl_metrics[-1], ctrl_timing=ctrl_timing[-1],
            pool_timing=pool_timing[-1],
            best_of_n={str(N): bon[N][-1] for N in N_VALUES})
        save_partial(partialp, a.n, N_VALUES, seeds_done)
        print(f"[seed {seed}] checkpointed", flush=True)

    # Costs are taken from THIS run's own timings, not from the audit. The
    # audit timed batches of 8; these run at --n, and the per-sonnet cost of
    # the controlled path depends strongly on batch size. Quoting audit
    # numbers next to results generated at a different batch size is what
    # produced the mismatched best-of-69 in the first place.
    per_sonnet_controlled = sum(t['wall_s'] for t in ctrl_timing) / len(ctrl_timing) / a.n
    per_sonnet_uncontrolled = (sum(p['wall_s'] for p in pool_timing)
                               / sum(p['pool_size'] for p in pool_timing))
    out = dict(
        n_per_seed=a.n, seeds=harness.SEEDS, selection='oracle-by-rhyme',
        n_forward_matched=n_fwd, n_wallclock_matched=n_wall,
        N_values=N_VALUES,
        budget_source=dict(
            wall_clock_budget_from=budget_src,
            ratio_forward_calls=audit['ratio_forward_calls'],
            measured_seconds_per_sonnet_controlled=round(per_sonnet_controlled, 3),
            measured_seconds_per_sonnet_uncontrolled=round(per_sonnet_uncontrolled, 4),
            measured_ratio_walltime=round(per_sonnet_controlled
                                          / per_sonnet_uncontrolled, 1)),
        pool_timing=pool_timing,
        controlled=dict(
            per_seed=[{k: round(r[k], 2) for k in harness.METRIC_KEYS}
                      for r in ctrl_metrics],
            aggregate=harness.aggregate_seeds(ctrl_metrics, harness.METRIC_KEYS),
            timing=ctrl_timing,
            seconds_per_sonnet=round(per_sonnet_controlled, 2)),
        best_of_n={
            str(N): dict(
                N=N,
                per_seed=[{k: round(r[k], 2) for k in harness.METRIC_KEYS}
                          for r in bon[N]],
                aggregate=harness.aggregate_seeds(bon[N], harness.METRIC_KEYS),
                seconds_per_sonnet=round(N * per_sonnet_uncontrolled, 2))
            for N in N_VALUES})

    outp = os.path.join(ROOT, 'results', 'best_of_n_results.json')
    with open(outp, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if os.path.exists(partialp):
        os.remove(partialp)   # final result written -- resume state is dead weight

    print(f"\n=== ORACLE BEST-OF-N SWEEP (n={a.n}, 3 semillas) ===")
    print(f"{'arm':<16} {'s/soneto':>9} {'Fmt%':>16} {'RimaFull%':>16}")
    for N in N_VALUES:
        e = out['best_of_n'][str(N)]
        f_ = e['aggregate']['format_pct']
        r_ = e['aggregate']['rhyme_full_pct']
        tag = f"best-of-{N}"
        if N == n_fwd:
            tag += "*"
        if N == n_wall:
            tag += "**"
        print(f"{tag:<16} {e['seconds_per_sonnet']:>9.2f} "
              f"{f_['mean']:>8.1f} +/-{f_['halfwidth']:<5.1f} "
              f"{r_['mean']:>8.1f} +/-{r_['halfwidth']:<5.1f}")
    c = out['controlled']['aggregate']
    print(f"{'CONTROLLED':<16} {out['controlled']['seconds_per_sonnet']:>9.2f} "
          f"{c['format_pct']['mean']:>8.1f} +/-{c['format_pct']['halfwidth']:<5.1f} "
          f"{c['rhyme_full_pct']['mean']:>8.1f} +/-{c['rhyme_full_pct']['halfwidth']:<5.1f}")
    print("  * forward-call matched   ** wall-clock matched")
    print("Guardado:", outp)


if __name__ == '__main__':
    main()
