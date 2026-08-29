"""Ablacion de respacing DDIM: 50 / 100 / 200 pasos. Mide formato/rima/silabas
y tiempo de muestreo. Checkpoint 50k EMA, sampler controlado (4 automaticos:
fmt 0.05, rhyme 0.5, rep 0.2, syll 0.05, K=5) -- sin concepto/emocion para
aislar la estructura. n=12 por config.

Uso: conda run -n gpu python resultados/experimentos/respacing_ablation.py [--n 12]
Salida: resultados/datos/respacing_results.json + tabla.
"""
import sys, os, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'vision2sonnet'))
sys.path.insert(0, os.path.join(ROOT, 'improved-diffusion'))

import torch
from functools import partial
from model_utils import load_diffusion_model, decode_to_text
from rhyme_es import build_suffix_matrix
import controllers_es
from eval_sonnets_es import compute_metrics
from improved_diffusion.test_util import denoised_fn_round

CKPT = os.path.join(ROOT, 'improved-diffusion/diffusion_models/'
                    'diff_e2e-tgt_pad_2000steps_16dim_sonnet_es/ema_0.9999_050000.pt')
RESPACINGS = ['ddim50', 'ddim100', 'ddim200']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    n = a.n

    results = []
    for resp in RESPACINGS:
        model, diffusion, model3, args, vocab = load_diffusion_model(CKPT, 'cuda', resp)
        id2word = {v: k for k, v in vocab.items()}
        eos_id = vocab['eos']
        S, _ = build_suffix_matrix(id2word, vocab_size=args.vocab_size)
        denoised_fn = partial(denoised_fn_round, args, model3.to('cuda'))
        shape = (n, args.image_size ** 2, args.in_channel)
        langevin_fn = controllers_es.make_langevin_fn(
            diffusion, model, S, eos_id,
            lambda_fmt=0.05, lambda_rhyme=0.5, lambda_rep=0.2, lambda_syll=0.05,
            K=5, vocab=vocab)

        t0 = time.time()
        with torch.no_grad():
            sample = diffusion.ddim_sample_loop(
                model, shape, clip_denoised=args.clip_denoised,
                denoised_fn=denoised_fn, model_kwargs={}, eta=1.0,
                top_p=-1.0, langevin_fn=langevin_fn)
        dt = time.time() - t0
        raw = decode_to_text(model, sample, vocab)
        m = compute_metrics(iter(raw))
        entry = dict(respacing=resp, steps=int(resp.replace('ddim', '')),
                     format_pct=round(m['format_pct'], 1),
                     rhyme_full_pct=round(m['rhyme_full_pct'], 1),
                     rhyme_quartet_pct=round(m['rhyme_quartet_pct'], 1),
                     syll_mean=round(m['syll_mean'], 1),
                     time_s=round(dt, 1), time_per_sonnet=round(dt / n, 1), n=n)
        results.append(entry)
        print(f"[{resp}] fmt={entry['format_pct']}% rhyfull={entry['rhyme_full_pct']}% "
              f"rhyquart={entry['rhyme_quartet_pct']}% {entry['time_per_sonnet']}s/soneto",
              flush=True)
        del model, diffusion, model3
        torch.cuda.empty_cache()

    out = os.path.join(ROOT, 'resultados', 'datos', 'respacing_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n=== ABLACION DE RESPACING (50k EMA, control, n=%d) ===" % n)
    print(f"{'Pasos':>6} {'Fmt%':>6} {'RimaFull%':>10} {'RimaCuart%':>11} {'s/soneto':>9}")
    for r in results:
        print(f"{r['steps']:>6} {r['format_pct']:>6} {r['rhyme_full_pct']:>10} "
              f"{r['rhyme_quartet_pct']:>11} {r['time_per_sonnet']:>9}")
    print("Guardado:", out)


if __name__ == '__main__':
    main()
