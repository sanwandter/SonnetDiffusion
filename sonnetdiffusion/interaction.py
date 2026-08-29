"""Dos checkpoints de interacción humano-computador (HCI) co-creativos.

HCI-1: validar/editar conceptos visuales extraídos de la imagen.
HCI-2: seleccionar una emoción de las top-5 candidatas.
"""
import torch
import torch.nn.functional as F


# =============================================================================
# ALGORITMO 5a: HCI-1 — Validación de conceptos visuales
# El usuario confirma, elimina o agrega conceptos. Se recalcula v_ref
# como mezcla ponderada (0.7·v_img + 0.3·media_conceptos).
# =============================================================================
def hci1_validate_concepts(concepts, v_img, encoder, input_fn=input, print_fn=print):
    kept = list(concepts)
    print_fn("\n=== HCI-1: Validación de Conceptos Visuales ===")
    while True:
        for i, (c, s) in enumerate(kept, 1):
            print_fn(f"  {i}. {c} ({s:.2f})")
        print_fn("Comandos: 'remove <concepto>', 'add <concepto>', 'done'")
        cmd = input_fn("> ").strip()
        if cmd == "done" or cmd == "":
            break
        if cmd.startswith("remove "):
            target = cmd[7:].strip()
            kept = [(c, s) for c, s in kept if c != target]
        elif cmd.startswith("add "):
            target = cmd[4:].strip()
            if target:
                kept.append((target, 1.0))

    if kept:
        cvecs = torch.stack([encoder.encode_text(c) for c, _ in kept]).mean(0)
        v_ref = F.normalize(0.7 * v_img + 0.3 * cvecs, dim=0)
    else:
        v_ref = v_img
    return kept, v_ref


# =============================================================================
# ALGORITMO 5b: HCI-2 — Selección de emoción
# El usuario elige una emoción de las top-N candidatas extraídas por SigLIP.
# =============================================================================
def hci2_select_emotion(emotions, concepts, encoder, input_fn=input, print_fn=print):
    print_fn("\n=== HCI-2: Selección de Emoción ===")
    concept_names = [c for c, _ in concepts]
    concept_vecs = torch.stack([encoder.encode_text(c) for c in concept_names])
    for i, (e, s) in enumerate(emotions, 1):
        e_vec = F.normalize(encoder.encode_text(e), dim=0)
        c_norm = F.normalize(concept_vecs, dim=1)
        sims = c_norm @ e_vec
        top_idx = sims.argsort(descending=True)[:5]
        support = ", ".join(concept_names[j] for j in top_idx if sims[j] > 0.02)
        print_fn(f"  {i}. {e} ({s:.2f})  [sustento: {support}]")
    while True:
        choice = input_fn("Elige (número): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(emotions):
            label = emotions[int(choice) - 1][0]
            return label, encoder.encode_text(label)
