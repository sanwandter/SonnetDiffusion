"""Shared harness for the DiffuLM experiments: seeding, generation, scoring, CIs.

Every experiment script in this directory is a thin driver over this module.
Nothing here re-implements sampling or metrics -- metrics always come from
sonnetdiffusion/eval_sonnets_es.py::compute_metrics.
"""
import json
import os
import random
import sys
import time
from functools import partial

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'sonnetdiffusion'))
sys.path.insert(0, os.path.join(ROOT, 'improved-diffusion'))

from eval_sonnets_es import compute_metrics, parse_sample, group_rhyme_counts
from rhyme_es import RHYME_GROUPS, count_syllables_verse

CKPT = os.path.join(ROOT, 'models', 'diffusion',
                    'diff_e2e-tgt_pad_2000steps_16dim_sonnet_es',
                    'ema_0.9999_050000.pt')

SEEDS = [102, 202, 302]

FULL = dict(lambda_fmt=0.05, lambda_rhyme=0.5, lambda_rep=0.2, lambda_syll=0.05,
            lambda_concept=0.15, lambda_emotion=0.15, lambda_repel=0.0, K=5)

CONCEPTS = ['pájaro', 'ave', 'ruiseñor', 'azul', 'pecado']
EMOTION = 'alegria'

METRIC_KEYS = ['format_pct', 'rhyme_full_pct', 'rhyme_quartet_pct',
               'syll_mean', 'syll_in_range_pct', 'distinct_endwords_pct']


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def set_seed(seed):
    """Seed every RNG the sampler touches.

    This controls the noise trajectory, which is what we want to vary across
    runs. It does not guarantee bitwise determinism across GPU kernels, and it
    does not need to -- we report variability, not bit-identical replays.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def mean_ci(values, confidence=0.95):
    """Mean and Student-t confidence interval over `values`."""
    from scipy import stats
    vals = list(values)
    if not vals:
        raise ValueError("mean_ci requires at least one value")
    n = len(vals)
    mean = float(np.mean(vals))
    if n == 1:
        halfwidth = 0.0
    else:
        sd = float(np.std(vals, ddof=1))
        tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
        halfwidth = tcrit * sd / np.sqrt(n)
    return dict(mean=mean, halfwidth=halfwidth,
                lo=mean - halfwidth, hi=mean + halfwidth, n=n)


def aggregate_seeds(per_seed, keys):
    """Map each metric key to a mean_ci dict computed across per-seed runs."""
    return {k: mean_ci([r[k] for r in per_seed]) for k in keys}


# --------------------------------------------------------------------------
# crash resume
# --------------------------------------------------------------------------
# These experiments run for hours on an exclusive GPU. Writing results only at
# the end means a reboot -- which has already cost this project ~50 minutes --
# destroys work that was fully computed. Every driver checkpoints after each
# unit of work instead, and skips units already on disk when restarted.

def load_partial(path, config):
    """Completed work units from a previous run, keyed by unit name.

    `config` identifies the parameters the results were produced under. A file
    written under different parameters is ignored rather than rejected, so
    changing --n does not require remembering to delete it -- the run simply
    starts over. Values must be JSON-native: a tuple saved becomes a list
    loaded, and would never compare equal.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        p = json.load(f)
    if p.get('config') != config:
        print(f"partial state is for a different config -- ignoring {path}")
        return {}
    return p.get('done', {})


def save_partial(path, config, done):
    """Checkpoint completed units. Atomic: a crash mid-write cannot corrupt."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dict(config=config, done=done), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def save_samples(path, lines, meta=None):
    """Write generated sample lines, one sonnet per line, atomically.

    The earlier experiments kept only aggregate metrics, so nothing could be
    rescored under a new metric without regenerating. Every new driver dumps
    its raw samples through this, with an optional JSON sidecar for timing.
    A .gz path is written compressed (the best-of-N pool is ~5k lines/seed).
    """
    import gzip
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opener = gzip.open if path.endswith('.gz') else open
    tmp = path + '.tmp'
    with opener(tmp, 'wt', encoding='utf-8') as f:
        for line in lines:
            f.write(line.replace('\n', ' ') + '\n')
    os.replace(tmp, path)
    if meta is not None:
        side = path[:-3] if path.endswith('.gz') else path
        side = os.path.splitext(side)[0] + '.json'
        with open(side, 'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)


def load_samples(path):
    """Read a file written by save_samples."""
    import gzip
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f if l.strip()]


def clear_partial(path):
    """Drop resume state once the real output file has been written."""
    if os.path.exists(path):
        os.remove(path)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_lines(lines):
    """Corpus-level metrics for a list of generated sample lines."""
    m = compute_metrics(iter(list(lines)))
    m.pop('_examples', None)
    return m


def score_one(line):
    """Per-sonnet metrics. Needed for oracle best-of-N selection."""
    verses, n_eos = parse_sample(line)
    rhyme = 0.0
    if n_eos == 14 and len(verses) >= 14:
        good, total = group_rhyme_counts(verses, RHYME_GROUPS)
        rhyme = good / total if total else 0.0
    sylls = [count_syllables_verse(v) for v in verses[:14] if v]
    sylls = [s for s in sylls if s > 0]
    return dict(n_lines=n_eos, rhyme=rhyme,
                syll_avg=(sum(sylls) / len(sylls)) if sylls else 0.0,
                syll_in_range=sum(1 for s in sylls if 10 <= s <= 12))


# --------------------------------------------------------------------------
# model + generation
# --------------------------------------------------------------------------
def load_ckpt(ckpt_path=CKPT, respacing='ddim100', device='cuda'):
    """Load model, diffusion, rounding weights, vocab and the suffix matrix.

    ckpt_path is parameterised so the checkpoint-progression experiment can
    sweep checkpoints without re-implementing this function.
    """
    from model_utils import load_diffusion_model
    from rhyme_es import build_suffix_matrix
    from improved_diffusion.test_util import denoised_fn_round

    model, diffusion, model3, args, vocab = load_diffusion_model(
        ckpt_path, device, respacing)
    id2word = {v: k for k, v in vocab.items()}
    S, _ = build_suffix_matrix(id2word, vocab_size=args.vocab_size)
    return dict(model=model, diffusion=diffusion, model3=model3, args=args,
                vocab=vocab, S=S, eos_id=vocab['eos'], device=device,
                denoised_fn=partial(denoised_fn_round, args, model3.to(device)))


def make_full_langevin(ctx, **overrides):
    """Build the 'full' controller stack, with per-lambda overrides."""
    import controllers_es
    from vocab_es import EMOTION_VOCAB

    cfg = dict(FULL)
    groups = overrides.pop('groups', None)   # rhyme scheme; default ABBA ABBA CDE CDE
    synalepha = overrides.pop('syll_synalepha', False)   # synalepha in the syllable penalty
    cfg.update(overrides)
    vocab = ctx['vocab']
    concept_ids = [vocab[w] for w in CONCEPTS if w in vocab]
    emotion_ids = [vocab[w] for w in EMOTION_VOCAB[EMOTION] if w in vocab]
    return controllers_es.make_langevin_fn(
        ctx['diffusion'], ctx['model'], ctx['S'], ctx['eos_id'],
        lambda_fmt=cfg['lambda_fmt'], lambda_rhyme=cfg['lambda_rhyme'],
        lambda_rep=cfg['lambda_rep'], lambda_syll=cfg['lambda_syll'],
        lambda_concept=cfg['lambda_concept'], lambda_emotion=cfg['lambda_emotion'],
        K=cfg['K'], vocab=vocab,
        concept_word_ids=concept_ids if cfg['lambda_concept'] > 0 else None,
        concept_weights={wid: 2.0 for wid in concept_ids},
        emotion_word_ids=emotion_ids if cfg['lambda_emotion'] > 0 else None,
        lambda_repel=cfg['lambda_repel'], syll_synalepha=synalepha,
        **({'groups': groups} if groups is not None else {}))


def generate(ctx, n, seed, langevin_fn):
    """Seeded generation. Returns (lines, wall_seconds, forward_calls).

    forward_calls counts denoiser forward passes, including the extra ones the
    Langevin inner loop triggers, giving a hardware-independent cost proxy
    alongside wall clock.
    """
    import torch
    from model_utils import decode_to_text

    model, args = ctx['model'], ctx['args']
    counter = {'n': 0}
    handle = model.register_forward_pre_hook(
        lambda m, inp: counter.__setitem__('n', counter['n'] + 1))
    try:
        set_seed(seed)
        shape = (n, args.image_size ** 2, args.in_channel)
        t0 = time.time()
        with torch.no_grad():
            sample = ctx['diffusion'].ddim_sample_loop(
                model, shape, clip_denoised=args.clip_denoised,
                denoised_fn=ctx['denoised_fn'], model_kwargs={}, eta=1.0,
                top_p=-1.0, langevin_fn=langevin_fn)
        dt = time.time() - t0
    finally:
        handle.remove()
    return decode_to_text(model, sample, ctx['vocab']), dt, counter['n']
