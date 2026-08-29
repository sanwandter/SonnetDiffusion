"""Traza la evolucion de un soneto a traves de los pasos de difusion DDIM,
con los 6 controladores Langevin activos (config grid-selected: co-creativa).

Replica el mecanismo de vision2sonnet/tools/demo_gradio.py (_run_generation +
_decode_live) pero fuera de Gradio: guarda el texto decodificado de
pred_xstart en varios pasos del muestreo para mostrar como el soneto pasa
de ruido a verso bajo guia analitica.

Uso: conda run -n gpu python resultados/experimentos/diffusion_evolution.py
Salida: resultados/datos/diffusion_evolution.json
"""
import sys, os, re, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision2sonnet"))
sys.path.insert(0, os.path.join(ROOT, "improved-diffusion"))

import torch
from functools import partial
from model_utils import load_diffusion_model, find_best_ckpt
from rhyme_es import build_suffix_matrix, rhyme_suffix
from vocab_es import EMOTION_VOCAB
import controllers_es
from improved_diffusion.test_util import denoised_fn_round

CONCEPTS = ["pájaro", "ave", "ruiseñor", "azul", "pecado"]
EMOTION = "alegria"

# Config full = grid-selected (co-creative), igual a ablation_controllers.py FULL.
FULL = dict(lambda_fmt=0.05, lambda_rhyme=0.5, lambda_rep=0.2, lambda_syll=0.05,
            lambda_concept=0.15, lambda_emotion=0.15, K=5)

_EOS_RE = re.compile(r"<\s*eos\s*>")


def _decode_pred_xstart_batch(model, sample, vocab):
    """Decodifica todo el batch; devuelve una lista de textos (uno por muestra)."""
    id2word = {v: k for k, v in vocab.items()}
    drop = {",", ".", ":", ";", "UNK", "PAD", "START", "END"}
    drop_ids = {vocab[t] for t in drop if t in vocab}
    with torch.no_grad():
        logits = model.get_logits(sample)
        ids = logits.argmax(dim=-1).cpu()
    out = []
    for seq in ids:
        words = [id2word.get(i.item(), "") for i in seq if i.item() not in drop_ids]
        verses = [p.strip() for p in _EOS_RE.split(" ".join(words)) if p.strip()]
        out.append("\n".join(verses))
    return out


_CONCEPT_RE = {
    w: re.compile(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)")
    for w in CONCEPTS
}


def _concepts_in(text):
    """Palabra completa exacta (sin substrings: 'azul' no matchea 'azuloso',
    'ave' no matchea 'suave')."""
    low = text.lower()
    return [w for w, pat in _CONCEPT_RE.items() if pat.search(low)]


def _abba_metrics(text):
    """Rima y conceptos restringidos al primer cuarteto (lineas 1-4, patron ABBA).

    rhyme_abba=True solo si ambos pares (1,4) y (2,3) comparten sufijo fonetico
    valido (rima completa, no parcial). concepts_abba: solo palabras completas
    presentes dentro de esas 4 lineas (no en el resto del soneto).
    """
    verses = [v for v in text.split("\n") if v.strip()]
    if len(verses) < 4:
        return dict(rhyme_abba=False, concepts_abba=[])
    quatrain = verses[:4]
    end = [v.split()[-1] if v.split() else "" for v in quatrain]
    s1, s2, s3, s4 = (rhyme_suffix(w) for w in end)
    rhyme_abba = bool(s1 and s4 and s1 == s4 and s2 and s3 and s2 == s3)
    concepts_abba = _concepts_in(" ".join(quatrain))
    return dict(rhyme_abba=rhyme_abba, concepts_abba=concepts_abba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--respacing", default="ddim100")
    ap.add_argument("--snapshots", type=int, default=6,
                     help="numero de snapshots espaciados uniformemente sobre el total de pasos")
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--n", type=int, default=12)
    args_cli = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args_cli.seed)

    ckpt = find_best_ckpt()
    model, diffusion, model3, args, vocab = load_diffusion_model(ckpt, device, args_cli.respacing)

    id2word = {v: k for k, v in vocab.items()}
    eos_id = vocab["eos"]
    S, _ = build_suffix_matrix(id2word, vocab_size=args.vocab_size)

    concept_ids = [vocab[w] for w in CONCEPTS if w in vocab]
    concept_weights = {wid: 2.0 for wid in concept_ids}
    emotion_ids = [vocab[w] for w in EMOTION_VOCAB[EMOTION] if w in vocab]

    langevin_fn = controllers_es.make_langevin_fn(
        diffusion, model, S, eos_id, vocab=vocab,
        concept_word_ids=concept_ids, concept_weights=concept_weights,
        emotion_word_ids=emotion_ids,
        lambda_fmt=FULL["lambda_fmt"], lambda_rhyme=FULL["lambda_rhyme"],
        lambda_rep=FULL["lambda_rep"], lambda_syll=FULL["lambda_syll"],
        lambda_concept=FULL["lambda_concept"], lambda_emotion=FULL["lambda_emotion"],
        K=FULL["K"],
    )

    denoised_fn = partial(denoised_fn_round, args, model3.to(device))
    n = args_cli.n
    shape = (n, args.image_size ** 2, args.in_channel)
    total = diffusion.num_timesteps

    snap_steps = sorted(set(
        max(1, round(total * f)) for f in
        [i / (args_cli.snapshots - 1) for i in range(args_cli.snapshots)]
    ))
    print(f"n={n} samples. Total steps: {total}. Snapshots at: {snap_steps}", flush=True)

    sample_iter = diffusion.ddim_sample_loop_progressive(
        model, shape, clip_denoised=args.clip_denoised, denoised_fn=denoised_fn,
        model_kwargs={}, eta=1.0, langevin_fn=langevin_fn)

    # traces[j] = lista de snapshots {step,total,text} para la muestra j
    traces = [[] for _ in range(n)]
    final_step = snap_steps[-1]
    out = None
    for i, out in enumerate(sample_iter):
        step_num = i + 1
        if step_num in snap_steps:
            # Igual que demo_gradio.py: el paso final se decodifica desde
            # out["sample"] (salida real tras clip+round), no desde pred_xstart
            # (que es solo la vista previa en vivo durante el loop).
            src = out["sample"] if step_num == final_step else out.get("pred_xstart", out["sample"])
            texts = _decode_pred_xstart_batch(model, src, vocab)
            for j, text in enumerate(texts):
                traces[j].append({"step": step_num, "total": total, "text": text})
            print(f"--- step {step_num}/{total} (sample 0) ---\n{texts[0]}\n", flush=True)

    last_step = snap_steps[-1]
    mid_step = snap_steps[len(snap_steps) // 2] if len(snap_steps) >= 3 else snap_steps[1]
    early_step = snap_steps[1] if len(snap_steps) > 1 else snap_steps[0]

    print(f"\nSeleccion (ABBA-scoped, palabra completa): early={early_step} "
          f"mid={mid_step} final={last_step}", flush=True)
    print(f"{'idx':<5}{'cpt@early':<11}{'cpt@mid':<10}{'cpt@final':<11}{'ABBA full@final':<16}")
    candidates = []
    for j in range(n):
        by_step = {snap["step"]: snap["text"] for snap in traces[j]}
        a_early = _abba_metrics(by_step[early_step])
        a_mid = _abba_metrics(by_step[mid_step])
        a_final = _abba_metrics(by_step[last_step])
        print(f"{j:<5}{len(a_early['concepts_abba']):<11}{len(a_mid['concepts_abba']):<10}"
              f"{len(a_final['concepts_abba']):<11}{str(a_final['rhyme_abba']):<16}")
        ok = (len(a_early["concepts_abba"]) == 0
              and len(a_mid["concepts_abba"]) < len(a_final["concepts_abba"])
              and a_final["rhyme_abba"])
        if ok:
            candidates.append((j, len(a_final["concepts_abba"])))

    if candidates:
        # entre quienes cumplen los 3 criterios, preferimos mas conceptos_abba@final
        candidates.sort(key=lambda c: c[1], reverse=True)
        selected = candidates[0][0]
        print(f"\nCandidatos que cumplen los 3 criterios: {[c[0] for c in candidates]}")
        print(f"Muestra seleccionada: {selected} "
              f"(concepts_abba@final={candidates[0][1]}, rhyme_abba@final=True)")
    else:
        # fallback: la que mas se acerque (rima ABBA completa, conceptos crecientes)
        def score(j):
            by_step = {snap["step"]: snap["text"] for snap in traces[j]}
            a_early = _abba_metrics(by_step[early_step])
            a_mid = _abba_metrics(by_step[mid_step])
            a_final = _abba_metrics(by_step[last_step])
            grows = len(a_mid["concepts_abba"]) < len(a_final["concepts_abba"])
            return (len(a_early["concepts_abba"]) == 0, grows, a_final["rhyme_abba"],
                    len(a_final["concepts_abba"]))
        selected = max(range(n), key=score)
        print(f"\nNinguna muestra cumple los 3 criterios exactamente; "
              f"usando la mas cercana: {selected}")

    result = {
        "checkpoint": os.path.basename(ckpt),
        "respacing": args_cli.respacing,
        "concepts": CONCEPTS,
        "emotion": EMOTION,
        "controller_config": FULL,
        "seed": args_cli.seed,
        "n_generated": n,
        "selection_criteria": {
            "early_step": early_step, "mid_step": mid_step, "final_step": last_step,
            "rule": "whole-word concepts within the first quatrain (lines 1-4) only: "
                    "0 at early, fewer at mid than at final, and full ABBA rhyme "
                    "(rhyme_suffix(line1)==rhyme_suffix(line4) and "
                    "rhyme_suffix(line2)==rhyme_suffix(line3)) at final",
        },
        "selected_index": selected,
        "trace": traces[selected],
    }
    out_path = os.path.join(ROOT, "resultados", "datos", "diffusion_evolution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Guardado:", out_path)


if __name__ == "__main__":
    main()
