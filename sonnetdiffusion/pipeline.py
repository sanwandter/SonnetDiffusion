"""Pipeline end-to-end Vision2Sonnet: imagen → soneto español.

Uso:
  python vision2sonnet/pipeline.py <imagen.jpg>
  python vision2sonnet/pipeline.py <imagen.jpg> --ckpt model050000.pt
  python vision2sonnet/pipeline.py <imagen.jpg> --device cpu
"""
import argparse
import json
import os
import re
import sys
from functools import partial

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIFF = os.path.join(ROOT, "improved-diffusion")
sys.path.insert(0, HERE)
sys.path.insert(0, DIFF)

from model_utils import load_diffusion_model, decode_to_text, find_best_ckpt, CKPT_DIR
from encoder import SiglipEncoder
from interaction import hci1_validate_concepts, hci2_select_emotion
from eval_sonnets_es import group_rhyme_counts
from rhyme_es import RHYME_GROUPS, last_rhyme_word, rhyme_suffix
from vocab_es import pysent_score_for_emotion


# =============================================================================
# ALGORITMO 6 (generación): Muestreo controlado DDIM + Langevin
# Genera N sonetos con controladores de formato, rima y anti-repetición.
# =============================================================================
def generate_sonnet(ckpt_path, device, n=12, respacing="ddim100",
                    lambda_fmt=0.05, lambda_rhyme=0.5, lambda_rep=0.2,
                    step_size=0.03, K=5, t_stop=10,
                    concept_words=None, emotion_label=None):
    from improved_diffusion.test_util import denoised_fn_round
    from rhyme_es import build_suffix_matrix
    from vocab_es import EMOTION_VOCAB
    import controllers_es

    model, diffusion, model3, args, vocab = load_diffusion_model(
        ckpt_path, device, respacing)

    id2word = {v: k for k, v in vocab.items()}
    eos_id = vocab["eos"]
    S, _ = build_suffix_matrix(id2word, vocab_size=args.vocab_size)

    concept_word_ids = None
    concept_neighborhoods = {}
    concept_weights = None
    if concept_words:
        concept_word_ids = [vocab[w] for w in concept_words if w in vocab]
        concept_weights = {wid: 2.0 for wid in concept_word_ids}
        concept_neighborhoods = {w: [] for w in concept_words if w in vocab}
        if not concept_word_ids:
            concept_word_ids = None

    emotion_word_ids = None
    if emotion_label and emotion_label in EMOTION_VOCAB:
        emotion_word_ids = [vocab[w] for w in EMOTION_VOCAB[emotion_label] if w in vocab]
        if not emotion_word_ids:
            emotion_word_ids = None

    langevin_fn = controllers_es.make_langevin_fn(
        diffusion, model, S, eos_id,
        lambda_fmt=lambda_fmt, lambda_rhyme=lambda_rhyme,
        lambda_rep=lambda_rep, step_size=step_size, K=K, t_stop=t_stop,
        vocab=vocab,
        concept_word_ids=concept_word_ids, concept_weights=concept_weights,
        emotion_word_ids=emotion_word_ids)

    shape = (n, args.image_size ** 2, args.in_channel)
    denoised_fn = partial(denoised_fn_round, args, model3.to(device))

    sample = diffusion.ddim_sample_loop(
        model, shape, clip_denoised=args.clip_denoised,
        denoised_fn=denoised_fn, model_kwargs={}, eta=1.0,
        top_p=-1.0, langevin_fn=langevin_fn)

    raw_lines = decode_to_text(model, sample, vocab)
    eos_re = re.compile(r"<\s*eos\s*>")
    sonnets = []
    for line in raw_lines:
        parts = eos_re.split(line)
        verses = [p.strip() for p in parts if p.strip()][:14]
        sonnets.append("\n".join(verses))
    return sonnets, concept_neighborhoods


# =============================================================================
# ALGORITMO 7: Evaluación compuesta E(c, img, e) = α·Rima + β·Visual + γ·Emo
# =============================================================================

def rhyme_accuracy(sonnet_text):
    """Precisión de rima por pares dentro de cada grupo ABBA ABBA CDE CDE."""
    verses = sonnet_text.split("\n")
    if len(verses) != 14:
        return 0.0
    verse_tokens = [v.split() for v in verses]
    good, total = group_rhyme_counts(verse_tokens, RHYME_GROUPS)
    return good / total if total else 0.0


def visual_coherence(sonnet_text, v_img, encoder):
    """Coherencia visual: similitud coseno promedio verso-imagen via SigLIP."""
    verses = sonnet_text.split("\n")
    vn = F.normalize(v_img, dim=0)
    sims = []
    for v in verses:
        e = F.normalize(encoder.encode_text(v), dim=0)
        sims.append(float(e @ vn))
    raw = sum(sims) / len(sims)
    return max(0.0, min(1.0, (raw + 1) / 2))


def emotional_alignment(sonnet_text, target_emotion, analyzer):
    """Alineación emocional: probabilidad pysentimiento mapeada a la emoción objetivo."""
    verses = sonnet_text.split("\n")
    scores = []
    for v in verses:
        out = analyzer.predict(v)
        scores.append(pysent_score_for_emotion(out.probas, target_emotion))
    return sum(scores) / len(scores) if scores else 0.0


def composite_score(rhyme, visual, emotional, alpha=0.4, beta=0.35, gamma=0.25):
    return alpha * rhyme + beta * visual + gamma * emotional


def evaluate_sonnet(sonnet_text, v_img, target_emotion, encoder, analyzer,
                    alpha=0.4, beta=0.35, gamma=0.25):
    r = rhyme_accuracy(sonnet_text)
    vis = visual_coherence(sonnet_text, v_img, encoder)
    emo = emotional_alignment(sonnet_text, target_emotion, analyzer)
    score = composite_score(r, vis, emo, alpha, beta, gamma)
    return {"rhyme": r, "visual": vis, "emotional": emo, "score": score}


# =============================================================================
# ALGORITMO 1: Pipeline end-to-end  imagen → soneto
# Orquesta: encoding (Alg 2-4) → HCI (Alg 5) → generación (Alg 6) → eval (Alg 7)
# =============================================================================

def run(image_path, ckpt_path=None, device="cuda", n_samples=12):
    from pysentimiento import create_analyzer

    # Lectura del input: validar la imagen antes de cargar modelos pesados.
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No existe la imagen de entrada: {image_path}")

    print("Cargando SigLIP 2...")
    enc = SiglipEncoder(device=device)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"No se pudo abrir '{image_path}' como imagen RGB: {e}")
    v_img = enc.encode_image(image)

    concepts = enc.extract_visual_concepts(v_img)
    emotions = enc.extract_emotion_candidates(v_img)

    kept, v_ref = hci1_validate_concepts(concepts, v_img, enc)
    emotion_label, v_emo = hci2_select_emotion(emotions, kept, enc)

    # HCI-1 (conceptos) y HCI-2 (emocion) guian la generacion.
    concept_words = [c for c, _ in kept]

    if ckpt_path is None:
        ckpt_path = find_best_ckpt()
    print(f"\nGenerando {n_samples} sonetos con {os.path.basename(ckpt_path)}...")
    sonnets, _ = generate_sonnet(ckpt_path, device, n=n_samples,
                                concept_words=concept_words,
                                emotion_label=emotion_label)

    print("\nCargando evaluador emocional...")
    analyzer = create_analyzer(task="emotion", lang="es")

    best_score = -1
    best_sonnet = None
    best_metrics = None
    for i, s in enumerate(sonnets):
        metrics = evaluate_sonnet(s, v_ref, emotion_label, enc, analyzer)
        tag = ""
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_sonnet = s
            best_metrics = metrics
            tag = " *"
        print(f"  [{i+1:2d}] R={metrics['rhyme']:.2f} V={metrics['visual']:.2f} "
              f"E={metrics['emotional']:.2f} -> {metrics['score']:.2f}{tag}")

    print("\n" + "=" * 50)
    print(f"Emoción: {emotion_label}")
    print(f"Conceptos: {', '.join(c for c, _ in kept[:5])}")
    print("=" * 50)
    print(best_sonnet)
    print("=" * 50)
    m = best_metrics
    print(f"Rima: {m['rhyme']:.2f}  Visual: {m['visual']:.2f}  "
          f"Emoción: {m['emotional']:.2f}  ->  Score: {m['score']:.2f}")

    return best_sonnet, best_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vision2Sonnet pipeline")
    ap.add_argument("image", help="Ruta a la fotografía de entrada")
    ap.add_argument("--ckpt", default=None, help="Ruta al checkpoint (default: último EMA)")
    ap.add_argument("--device", default="cuda", help="cpu o cuda")
    ap.add_argument("--n", type=int, default=12, help="Número de muestras a generar")
    args = ap.parse_args()
    run(args.image, ckpt_path=args.ckpt, device=args.device, n_samples=args.n)
