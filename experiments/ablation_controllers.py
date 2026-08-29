"""Ablacion por controlador: desactiva un controlador Langevin a la vez (lambda=0)
y mide formato/rima/conceptos/silabas. Checkpoint 50k EMA, config full = la
seleccionada en el grid (lambda_rhy=0.5, cpt=0.3, emo=0.15, K=5).

Uso: conda run -n gpu python resultados/experimentos/ablation_controllers.py [--n 6]
Salida: resultados/datos/ablation_results.json + tabla por consola.
"""
import sys, os, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'vision2sonnet'))
sys.path.insert(0, os.path.join(ROOT, 'improved-diffusion'))

import torch
from functools import partial
from model_utils import load_diffusion_model, decode_to_text
from rhyme_es import build_suffix_matrix, count_syllables_verse
from vocab_es import EMOTION_VOCAB
import controllers_es
from eval_sonnets_es import parse_sample, group_rhyme_counts
from rhyme_es import RHYME_GROUPS
from improved_diffusion.test_util import denoised_fn_round


def eval_sonnet(line, concept_words, vocab, args):
    verses, n_eos = parse_sample(line)
    n_lines = n_eos
    rhyme = 0.0
    if n_lines == 14 and len(verses) >= 14:
        good, total = group_rhyme_counts(verses, RHYME_GROUPS)
        rhyme = good / total if total else 0.0
    found = []
    text_lower = line.lower()
    for w in concept_words:
        if w.lower() in text_lower:
            found.append(w)
    sylls = [count_syllables_verse(v) for v in verses[:14] if v]
    sylls = [s for s in sylls if s > 0]
    syll_avg = sum(sylls) / len(sylls) if sylls else 0.0
    syll_in_range = sum(1 for s in sylls if 10 <= s <= 12)
    return dict(text=line, n_lines=n_lines, rhyme=rhyme,
                concepts_present=len(found), concepts_found=found,
                syll_avg=round(syll_avg, 1), syll_in_range=syll_in_range,
                syllables=sylls)

CKPT = os.path.join(ROOT, 'improved-diffusion/diffusion_models/'
                    'diff_e2e-tgt_pad_2000steps_16dim_sonnet_es/ema_0.9999_050000.pt')
CONCEPTS = ['pájaro', 'ave', 'ruiseñor', 'azul', 'pecado']
EMOTION = 'alegria'

# Config full = seleccionada en el grid search.
FULL = dict(lambda_fmt=0.05, lambda_rhyme=0.5, lambda_rep=0.2, lambda_syll=0.05,
            lambda_concept=0.15, lambda_emotion=0.15, K=5)

# 8 configuraciones de ablacion. None = sin control (langevin_fn=None).
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


def summarize(sonnets):
    n = len(sonnets)
    fmt = sum(1 for s in sonnets if s['n_lines'] == 14) / n * 100
    rhy = [s['rhyme'] for s in sonnets if s['n_lines'] == 14]
    rhy_pct = (sum(rhy) / len(rhy) * 100) if rhy else 0.0
    cpt = sum(s['concepts_present'] for s in sonnets) / n
    syl = sum(s['syll_avg'] for s in sonnets) / n
    syl_rng = sum(s['syll_in_range'] for s in sonnets) / n
    return dict(format_pct=round(fmt, 1), rhyme_pct=round(rhy_pct, 1),
                concepts=round(cpt, 2), syll_avg=round(syl, 1),
                syll_in_range=round(syl_rng, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    args_cli = ap.parse_args()
    n = args_cli.n

    model, diffusion, model3, args, vocab = load_diffusion_model(CKPT, 'cuda', 'ddim100')
    id2word = {v: k for k, v in vocab.items()}
    eos_id = vocab['eos']
    S, _ = build_suffix_matrix(id2word, vocab_size=args.vocab_size)
    denoised_fn = partial(denoised_fn_round, args, model3.to('cuda'))
    shape = (n, args.image_size ** 2, args.in_channel)

    concept_ids = [vocab[w] for w in CONCEPTS if w in vocab]
    concept_weights = {wid: 2.0 for wid in concept_ids}
    emotion_ids = [vocab[w] for w in EMOTION_VOCAB[EMOTION] if w in vocab]

    results = []
    for name, override in ABLATIONS:
        t0 = time.time()
        print(f"[{name}] generating n={n} ...", flush=True)
        if override is None:
            langevin_fn = None
        else:
            cfg = dict(FULL); cfg.update(override)
            emo_ids = emotion_ids if cfg['lambda_emotion'] > 0 else None
            cpt_ids = concept_ids if cfg['lambda_concept'] > 0 else None
            langevin_fn = controllers_es.make_langevin_fn(
                diffusion, model, S, eos_id,
                lambda_fmt=cfg['lambda_fmt'], lambda_rhyme=cfg['lambda_rhyme'],
                lambda_rep=cfg['lambda_rep'], lambda_syll=cfg['lambda_syll'],
                lambda_concept=cfg['lambda_concept'], lambda_emotion=cfg['lambda_emotion'],
                K=cfg['K'], vocab=vocab,
                concept_word_ids=cpt_ids, concept_weights=concept_weights,
                emotion_word_ids=emo_ids)

        with torch.no_grad():
            sample = diffusion.ddim_sample_loop(
                model, shape, clip_denoised=args.clip_denoised,
                denoised_fn=denoised_fn, model_kwargs={}, eta=1.0,
                top_p=-1.0, langevin_fn=langevin_fn)
        raw = decode_to_text(model, sample, vocab)
        sonnets = [eval_sonnet(line, CONCEPTS, vocab, args) for line in raw]
        stats = summarize(sonnets)
        stats['config'] = name
        stats['time_s'] = round(time.time() - t0, 1)
        stats['sonnets'] = sonnets
        results.append(stats)
        print(f"  fmt={stats['format_pct']}% rhy={stats['rhyme_pct']}% "
              f"cpt={stats['concepts']}/5 syl={stats['syll_avg']} "
              f"({stats['time_s']}s)", flush=True)

    out = os.path.join(ROOT, 'resultados', 'datos', 'ablation_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n=== ABLACION POR CONTROLADOR (50k EMA, n=%d) ===" % n)
    print(f"{'Config':<12} {'Fmt%':>6} {'Rima%':>6} {'Cpt/5':>6} {'Syl':>5} {'SylRng':>7}")
    for r in results:
        print(f"{r['config']:<12} {r['format_pct']:>6} {r['rhyme_pct']:>6} "
              f"{r['concepts']:>6} {r['syll_avg']:>5} {r['syll_in_range']:>7}")
    print("Guardado:", out)


if __name__ == '__main__':
    main()
