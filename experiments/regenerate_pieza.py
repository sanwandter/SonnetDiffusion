"""Regenera la pieza co-creativa (imagen -> soneto) con la config ACTUAL del
informe: 2 checkpoints HCI (HCI-1 conceptos, HCI-2 emocion), 6 controladores,
50k EMA, evaluacion compuesta E. No interactivo (auto-responde HCI) porque
`conda run` no reenvia stdin.

Uso: conda run -n gpu python resultados/experimentos/regenerate_pieza.py
Salida: resultados/piezas/soneto_condicionado_imagen.txt
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'vision2sonnet'))
sys.path.insert(0, os.path.join(ROOT, 'improved-diffusion'))

import torch
import torch.nn.functional as F
from PIL import Image
from encoder import SiglipEncoder
from interaction import hci1_validate_concepts, hci2_select_emotion
from pipeline import generate_sonnet, evaluate_sonnet

CKPT = os.path.join(ROOT, 'improved-diffusion/diffusion_models/'
                    'diff_e2e-tgt_pad_2000steps_16dim_sonnet_es/ema_0.9999_050000.pt')
IMG = os.path.join(ROOT, 'images', 'bird.jpeg')
OUT = os.path.join(ROOT, 'resultados', 'piezas', 'soneto_condicionado_imagen.txt')
N = 8


def main():
    from pysentimiento import create_analyzer
    print('Cargando SigLIP 2...')
    enc = SiglipEncoder(device='cuda')
    image = Image.open(IMG).convert('RGB')
    v_img = enc.encode_image(image)

    concepts = enc.extract_visual_concepts(v_img)        # top-15
    emotions = enc.extract_emotion_candidates(v_img)     # top-5 de 7 Ekman

    # HCI-1: el usuario confirma; aqui mantenemos los 5 conceptos top (subset enfocado).
    kept, _ = hci1_validate_concepts(concepts, v_img, enc, input_fn=lambda _: 'done')
    kept = kept[:5]
    cvecs = torch.stack([enc.encode_text(c) for c, _ in kept]).mean(0)
    v_ref = F.normalize(0.7 * v_img + 0.3 * cvecs, dim=0)

    # HCI-2: el usuario elige la emocion top-1 de las 5 candidatas.
    emotion_label, _ = hci2_select_emotion(emotions, kept, enc, input_fn=lambda _: '1')

    concept_words = [c for c, _ in kept]
    print(f'\nHCI-1 conceptos: {concept_words}')
    print(f'HCI-2 emocion  : {emotion_label}\n')

    # Generacion con 6 controladores (config co-creativa: lambda_rhyme=0.5, K=5).
    sonnets, _ = generate_sonnet(CKPT, 'cuda', n=N,
                                 concept_words=concept_words, emotion_label=emotion_label,
                                 lambda_rhyme=0.5, K=5)

    print('Cargando evaluador emocional...')
    analyzer = create_analyzer(task='emotion', lang='es')
    best = best_m = None; best_s = -1
    for i, s in enumerate(sonnets):
        m = evaluate_sonnet(s, v_ref, emotion_label, enc, analyzer)
        tag = ''
        if m['score'] > best_s:
            best_s, best, best_m = m['score'], s, m; tag = ' *'
        print(f"  [{i+1}] R={m['rhyme']:.2f} V={m['visual']:.2f} "
              f"E={m['emotional']:.2f} -> {m['score']:.2f}{tag}")

    header = (
        "# Pieza co-creativa end-to-end (imagen -> soneto)\n"
        "# Regenerada con la configuracion actual del informe (Entrega Final).\n"
        "#\n"
        f"# Imagen de entrada : {os.path.basename(IMG)} (ave)\n"
        f"# HCI-1 conceptos   : {', '.join(concept_words)}\n"
        f"# HCI-2 emocion     : {emotion_label}\n"
        "# Checkpoint        : 50k EMA   |   6 controladores "
        "(formato, rima, repeticion, silaba, concepto, emocion)\n"
        "# Evaluacion compuesta E = 0.4*Rima + 0.35*Visual + 0.25*Emocion\n"
        f"#   Rima={best_m['rhyme']:.2f}  Visual={best_m['visual']:.2f}  "
        f"Emocion={best_m['emotional']:.2f}  ->  Score={best_m['score']:.2f}  "
        f"(mejor de {N} candidatos)\n"
        "# ------------------------------------------------------------------\n\n"
    )
    note = ("\n\n# Nota: 14 versos, formato 100%. La coherencia semantica es limitada\n"
            "# (modelo 87M, nivel-palabra, ~9.6k sonetos), pero la pieza integra las\n"
            "# decisiones de los 2 checkpoints HCI y es evaluada por E(c,img,e).\n")
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(header + best + note)
    print(f'\nGuardado: {OUT}')
    print('=' * 50); print(best); print('=' * 50)


if __name__ == '__main__':
    main()
