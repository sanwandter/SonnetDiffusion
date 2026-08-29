"""Compara calidad de sonetos entre checkpoints y configuraciones de hiperparámetros.

Uso:
  python vision2sonnet/tools/compare_checkpoints.py --steps 10 20 30 40 50 60
  python vision2sonnet/tools/compare_checkpoints.py --steps 60 --sweep
  python vision2sonnet/tools/compare_checkpoints.py --all --n 6
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V2S = os.path.dirname(HERE)
ROOT = os.path.dirname(V2S)
sys.path.insert(0, V2S)
sys.path.insert(0, os.path.join(ROOT, "improved-diffusion"))

from model_utils import CKPT_DIR, find_ckpt, all_checkpoint_steps, load_diffusion_model, decode_to_text
from eval_sonnets_es import compute_metrics
from rhyme_es import build_suffix_matrix

RESULTS_PATH = os.path.join(V2S, "checkpoint_results.json")


def generate_and_eval(ckpt_path, device, n, controlled, hparams,
                      concept_words=None, emotion_label=None):
    """Genera n sonetos y calcula métricas de formato, rima y sílabas."""
    from functools import partial
    from improved_diffusion.test_util import denoised_fn_round
    import controllers_es
    from vocab_es import EMOTION_VOCAB

    respacing = hparams.get("respacing", "ddim100") if controlled else ""
    model, diffusion, model3, args, vocab = load_diffusion_model(
        ckpt_path, device, respacing)

    langevin_fn = None
    if controlled:
        id2word = {v: k for k, v in vocab.items()}
        S, _ = build_suffix_matrix(id2word, vocab_size=args.vocab_size)

        concept_word_ids = None
        concept_weights = None
        if concept_words:
            concept_word_ids = [vocab[w] for w in concept_words if w in vocab]
            concept_weights = {wid: 2.0 for wid in concept_word_ids}
            if not concept_word_ids:
                concept_word_ids = None

        emotion_word_ids = None
        if emotion_label and emotion_label in EMOTION_VOCAB:
            emotion_word_ids = [vocab[w] for w in EMOTION_VOCAB[emotion_label] if w in vocab]
            if not emotion_word_ids:
                emotion_word_ids = None

        langevin_fn = controllers_es.make_langevin_fn(
            diffusion, model, S, vocab["eos"],
            lambda_fmt=hparams.get("lambda_fmt", 0.05),
            lambda_rhyme=hparams.get("lambda_rhyme", 0.5),
            lambda_rep=hparams.get("lambda_rep", 0.2),
            lambda_syll=hparams.get("lambda_syll", 0.05),
            step_size=hparams.get("step_size", 0.03),
            K=hparams.get("K", 5),
            t_stop=hparams.get("t_stop", 10),
            vocab=vocab,
            concept_word_ids=concept_word_ids,
            concept_weights=concept_weights,
            lambda_concept=hparams.get("lambda_concept", 0.15),
            emotion_word_ids=emotion_word_ids,
            lambda_emotion=hparams.get("lambda_emotion", 0.15))

    shape = (n, args.image_size ** 2, args.in_channel)
    denoised_fn = partial(denoised_fn_round, args, model3.to(device))

    t0 = time.time()
    if controlled:
        sample = diffusion.ddim_sample_loop(
            model, shape, clip_denoised=False,
            denoised_fn=denoised_fn, model_kwargs={}, eta=1.0,
            top_p=-1.0, langevin_fn=langevin_fn)
    else:
        sample = diffusion.p_sample_loop(
            model, shape, clip_denoised=False,
            denoised_fn=denoised_fn, model_kwargs={}, top_p=-1.0)
    gen_time = time.time() - t0

    raw_lines = decode_to_text(model, sample, vocab)
    metrics = compute_metrics(raw_lines)
    metrics["gen_time_s"] = round(gen_time, 1)

    import re
    eos_re = re.compile(r"<\s*eos\s*>")
    sonnets_text = []
    for line in raw_lines:
        parts = eos_re.split(line)
        verses = [p.strip() for p in parts if p.strip()][:14]
        sonnets_text.append("\n".join(verses))

    return metrics, sonnets_text


def print_table(results):
    has_sweep = any(r.get("hparams") not in ("none", "custom", None) for r in results)
    if has_sweep:
        header = (f"{'Step':>6s} {'Config':<12s} {'l_rhy':>5s} {'l_rep':>5s} "
                  f"{'Fmt%':>5s} {'RhyAll%':>7s} {'RhyQ%':>6s} "
                  f"{'Syll':>5s} {'Dist%':>5s} {'Time':>5s}")
    else:
        header = (f"{'Step':>6s} {'Modo':<5s} {'Fmt%':>5s} {'RhyAll%':>7s} {'RhyQ%':>6s} "
                  f"{'Syll':>5s} {'Dist%':>5s} {'Time':>5s}")
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        if has_sweep:
            hp = r.get("hparam_values", {})
            lr = f"{hp.get('lambda_rhyme', '-'):>5s}" if isinstance(hp.get('lambda_rhyme'), str) else f"{hp.get('lambda_rhyme', 0):>5.2f}"
            lrp = f"{hp.get('lambda_rep', '-'):>5s}" if isinstance(hp.get('lambda_rep'), str) else f"{hp.get('lambda_rep', 0):>5.2f}"
            name = r.get("hparams", "free")
            if not r["controlled"]:
                name = "free"
                lr = "  -  "
                lrp = "  -  "
            print(f"{r['step_k']*1000:>6d} {name:<12s} {lr} {lrp} "
                  f"{m['format_pct']:>5.1f} {m['rhyme_full_pct']:>7.1f} "
                  f"{m['rhyme_quartet_pct']:>6.1f} {m['syll_mean']:>5.1f} "
                  f"{m['distinct_endwords_pct']:>5.1f} {m['gen_time_s']:>5.1f}")
        else:
            mode = "ctrl" if r["controlled"] else "free"
            print(f"{r['step_k']*1000:>6d} {mode:<5s} {m['format_pct']:>5.1f} "
                  f"{m['rhyme_full_pct']:>7.1f} {m['rhyme_quartet_pct']:>6.1f} "
                  f"{m['syll_mean']:>5.1f} {m['distinct_endwords_pct']:>5.1f} "
                  f"{m['gen_time_s']:>5.1f}")


SWEEP_CONFIGS = [
    {"name": "optimal",     "lambda_fmt": 0.05, "lambda_rhyme": 0.5,  "lambda_rep": 0.2, "lambda_concept": 0.15, "lambda_emotion": 0.15, "lambda_syll": 0.05, "step_size": 0.03, "K": 5},
    {"name": "high_rhy",    "lambda_fmt": 0.05, "lambda_rhyme": 0.5,  "lambda_rep": 0.2, "lambda_concept": 0.3, "lambda_emotion": 0.3,  "lambda_syll": 0.05, "step_size": 0.03, "K": 7},
    {"name": "high_cpt",    "lambda_fmt": 0.05, "lambda_rhyme": 0.4,  "lambda_rep": 0.2, "lambda_concept": 0.8, "lambda_emotion": 0.15, "lambda_syll": 0.05, "step_size": 0.03, "K": 5},
    {"name": "gentle",      "lambda_fmt": 0.03, "lambda_rhyme": 0.2,  "lambda_rep": 0.1, "lambda_concept": 0.1, "lambda_emotion": 0.05, "lambda_syll": 0.03, "step_size": 0.02, "K": 3},
    {"name": "no_ctrl",     "lambda_fmt": 0.0,  "lambda_rhyme": 0.0,  "lambda_rep": 0.0, "lambda_concept": 0.0, "lambda_emotion": 0.0,  "lambda_syll": 0.0,  "step_size": 0.03, "K": 1},
]


def main():
    ap = argparse.ArgumentParser(description="Comparar checkpoints e hiperparámetros")
    ap.add_argument("--steps", type=int, nargs="+", help="Steps en miles (ej: 10 20 30)")
    ap.add_argument("--all", action="store_true", help="Evaluar todos los checkpoints")
    ap.add_argument("--n", type=int, default=8, help="Muestras por checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="store_true", help="Sweep de hiperparámetros")
    ap.add_argument("--lambda_fmt", type=float, default=0.05)
    ap.add_argument("--lambda_rhyme", type=float, default=0.5)
    ap.add_argument("--lambda_rep", type=float, default=0.2)
    ap.add_argument("--lambda_concept", type=float, default=0.15)
    ap.add_argument("--lambda_emotion", type=float, default=0.15)
    ap.add_argument("--lambda_syll", type=float, default=0.05)
    ap.add_argument("--step_size", type=float, default=0.03)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--concepts", type=str, nargs="*", default=None,
                    help="Palabras concepto a forzar (ej: pájaro ave azul)")
    ap.add_argument("--emotion", type=str, default=None,
                    help="Etiqueta emocional (ej: alegria, tristeza)")
    ap.add_argument("--save_sonnets", action="store_true", help="Guardar sonetos generados")
    args = ap.parse_args()

    if args.all:
        steps = all_checkpoint_steps()
    elif args.steps:
        steps = args.steps
    else:
        steps = [s for s in all_checkpoint_steps() if s % 10 == 0]

    hparams_list = SWEEP_CONFIGS if args.sweep else [
        {"name": "custom", "lambda_fmt": args.lambda_fmt,
         "lambda_rhyme": args.lambda_rhyme, "lambda_rep": args.lambda_rep,
         "lambda_concept": args.lambda_concept, "lambda_emotion": args.lambda_emotion,
         "lambda_syll": args.lambda_syll,
         "step_size": args.step_size, "K": args.K}
    ]

    all_results = []

    for step_k in steps:
        ckpt = find_ckpt(step_k)
        if ckpt is None:
            print(f"  saltando {step_k}k — sin checkpoint")
            continue

        print(f"\n{'='*60}")
        print(f"  Checkpoint {step_k*1000} — {os.path.basename(ckpt)}")
        print(f"{'='*60}")

        print(f"  Generando {args.n} sin control...")
        m_free, sonnets_free = generate_and_eval(ckpt, args.device, args.n, False, {})
        result_free = {"step_k": step_k, "controlled": False, "hparams": "none",
                       "metrics": m_free}
        all_results.append(result_free)

        if args.save_sonnets:
            out_dir = os.path.join(V2S, "checkpoint_sonnets")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"step{step_k*1000:06d}_free.txt"), "w") as f:
                f.write("\n\n---\n\n".join(sonnets_free))

        for hp in hparams_list:
            hp_name = hp.get("name", "custom")
            print(f"  Generando {args.n} controlado ({hp_name})...")
            m_ctrl, sonnets_ctrl = generate_and_eval(
                ckpt, args.device, args.n, True, hp,
                concept_words=args.concepts, emotion_label=args.emotion)
            result_ctrl = {"step_k": step_k, "controlled": True, "hparams": hp_name,
                           "metrics": m_ctrl,
                           "hparam_values": {k: v for k, v in hp.items() if k != "name"}}
            all_results.append(result_ctrl)

            if args.save_sonnets:
                with open(os.path.join(out_dir, f"step{step_k*1000:06d}_ctrl_{hp_name}.txt"), "w") as f:
                    f.write("\n\n---\n\n".join(sonnets_ctrl))

    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}\n")
    print_table(all_results)

    prev = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            prev = json.load(f)
    run_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": args.n,
        "results": all_results,
    }
    prev.append(run_entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"\nResultados guardados en {RESULTS_PATH}")

    best = max((r for r in all_results if r["controlled"]),
               key=lambda r: r["metrics"]["rhyme_full_pct"], default=None)
    if best:
        print(f"\nMejor controlado: step {best['step_k']*1000}, "
              f"hparams={best['hparams']}, "
              f"rima={best['metrics']['rhyme_full_pct']:.1f}%")


if __name__ == "__main__":
    main()
