"""Utilidades compartidas para cargar modelo de difusión y decodificar muestras.

Centraliza lógica duplicada en pipeline.py, compare_checkpoints.py
y sonnet_es_sample.py.
"""
import json
import os
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CKPT_DIR = os.path.join(ROOT, "models", "diffusion",
                        "diff_e2e-tgt_pad_2000steps_16dim_sonnet_es")


def find_ckpt(step_k):
    """Busca checkpoint EMA o raw por step en miles. Prefiere EMA si existe."""
    step = step_k * 1000
    s = f"{step:06d}"
    ema = os.path.join(CKPT_DIR, f"ema_0.9999_{s}.pt")
    raw = os.path.join(CKPT_DIR, f"model{s}.pt")
    if os.path.exists(ema):
        return ema
    if os.path.exists(raw):
        return raw
    return None


def all_checkpoint_steps():
    """Lista todos los steps (en miles) con checkpoint disponible."""
    pts = [f for f in os.listdir(CKPT_DIR) if f.startswith("model") and f.endswith(".pt")]
    return sorted(int(re.search(r"(\d+)", f).group(1)) // 1000 for f in pts)


def find_latest_ckpt():
    """Selecciona el último checkpoint EMA (≥50k) o modelo raw."""
    pts = sorted(f for f in os.listdir(CKPT_DIR)
                 if f.startswith("ema_0.9999_") and f.endswith(".pt"))
    if not pts:
        pts = sorted(f for f in os.listdir(CKPT_DIR)
                     if f.startswith("model") and f.endswith(".pt"))
    return os.path.join(CKPT_DIR, pts[-1])


# Mejor checkpoint: 50k EMA con controladores optimizados
# (rhy=0.5, cpt=0.3, emo=0.15, K=5): rima 93.3%, formato 100%.
BEST_STEP = 50000


def find_best_ckpt():
    """Mejor checkpoint (50k EMA); cae al ultimo si no existe en disco."""
    best = os.path.join(CKPT_DIR, f"ema_0.9999_{BEST_STEP:06d}.pt")
    return best if os.path.exists(best) else find_latest_ckpt()


def load_diffusion_model(ckpt_path, device, respacing="ddim100"):
    """Carga modelo de difusión, rounding weights y vocabulario.

    Returns: (model, diffusion, model3, args, vocab)
    """
    from improved_diffusion.rounding import load_models
    from improved_diffusion.test_util import get_weights
    from improved_diffusion.script_util import (
        model_and_diffusion_defaults, create_model_and_diffusion,
    )

    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, "training_args.json")) as f:
        training_args = json.load(f)

    defaults = model_and_diffusion_defaults()
    defaults.update(training_args)
    defaults["timestep_respacing"] = respacing
    defaults["sigma_small"] = True
    defaults.setdefault("clip_denoised", False)
    defaults.setdefault("modality", "e2e-tgt")
    defaults.setdefault("experiment", "random")
    defaults.setdefault("model_name_or_path", "")

    model, diffusion = create_model_and_diffusion(
        **{k: defaults[k] for k in model_and_diffusion_defaults().keys()})
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()

    model2, _ = load_models(
        defaults["modality"], defaults["experiment"],
        defaults.get("model_name_or_path", ""),
        defaults["in_channel"], ckpt_dir)
    if defaults.get("training_mode", "").startswith("e2e"):
        model2.weight = torch.nn.Parameter(model.word_embedding.weight.clone().cpu())

    class _Args:
        pass
    args = _Args()
    for k, v in defaults.items():
        setattr(args, k, v)
    model3 = get_weights(model2, args)

    vocab = json.load(open(os.path.join(ckpt_dir, "vocab.json")))
    return model, diffusion, model3, args, vocab


# Tokens a descartar en la decodificación
_DROP_TOKENS = {",", ".", ":", ";", "UNK", "PAD", "START", "END"}


def decode_to_text(model, sample, vocab):
    """Decodifica muestras de difusión a líneas de texto crudo.

    Usa lm_head (get_logits → argmax), filtra puntuación y tokens especiales.
    Mantiene < eos > intactos para que el evaluador cuente versos.
    """
    id2word = {v: k for k, v in vocab.items()}
    drop_ids = {vocab[t] for t in _DROP_TOKENS if t in vocab}

    with torch.no_grad():
        logits = model.get_logits(sample)
        ids = logits.argmax(dim=-1).cpu()

    lines = []
    for seq in ids:
        words = []
        for idx in seq:
            i = idx.item()
            if i not in drop_ids:
                words.append(id2word.get(i, ""))
        lines.append(" ".join(words))
    return lines
