"""Dream-7B-Instruct masked-DLM baseline.

This is a REFERENCE POINT, not a head-to-head. The comparison is unfair in
both directions: our diffusion model is trained in-domain on 8,648 Spanish
sonnets and is far smaller; Dream is prompted zero-shot and never saw the
corpus. What it establishes is what an off-the-shelf masked DLM produces on
this constraint class, and that its published control mechanisms
(classifier guidance, automata- or CFG-constrained decoding) do not apply to
rhyme and meter without extra machinery.

Generation is per-sonnet and resumable: the three output files are rewritten
after every sonnet, and a restart continues from however many are already on
disk rather than regenerating them.

Run in the `dream` env (transformers 4.46.2 / torch 2.5.1), NOT `gpu`:
  conda run -n dream python experiments/dream_baseline.py --n 30
Salida: results/dream_baseline/dream7b_sonnets.txt
"""
import argparse
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODEL_ID = 'Dream-org/Dream-v0-Instruct-7B'

PROMPT = (
    "Escribe un soneto en español. Debe tener exactamente 14 versos, "
    "cada verso de once sílabas (endecasílabo), con esquema de rima "
    "ABBA ABBA CDE CDE. Tema: {tema}. "
    "Responde únicamente con los 14 versos, uno por línea, sin título "
    "ni comentarios."
)

TEMAS = ['el mar', 'la noche', 'el amor', 'el tiempo', 'la muerte',
         'la ciudad', 'el olvido', 'la aurora', 'el destierro', 'la lluvia',
         'el invierno', 'la memoria', 'el silencio', 'la rosa', 'el viento',
         'la luna', 'el río', 'la ausencia', 'el fuego', 'la esperanza',
         'el sueño', 'la montaña', 'el otoño', 'la sombra', 'el espejo',
         'la infancia', 'el camino', 'la piedra', 'el árbol', 'la estrella']


def to_eval_line(text):
    """Convert Dream's free-form output into the one-line <eos> format that
    eval_sonnets_es.py parses. Applies the same normalisation as the corpus:
    lowercase, punctuation stripped, accents kept.

    Returns (eval_line, n_raw_verses). n_raw_verses is the count BEFORE
    truncation to 14 and is what the paper must report as Dream's format
    accuracy -- truncating first and then measuring format would score 100%
    for any model that emits at least 14 lines, which measures this function
    rather than the model.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    lines = [l for l in lines if not re.match(r'^\s*(soneto|título|tema)\b', l, re.I)]
    n_raw = len(lines)
    cleaned = []
    for l in lines[:14]:
        l = l.lower()
        l = ''.join(c for c in l
                    if c.isalpha() or c.isspace() or unicodedata.combining(c))
        l = re.sub(r'\s+', ' ', l).strip()
        if l:
            cleaned.append(l)
    return ' '.join(f"{l} <eos>" for l in cleaned), n_raw


def load_resume(raw_path, eval_path, counts_path):
    """Sonnets already generated, as (raws, evals, raw_counts).

    All three files are rewritten together after each sonnet, but a crash
    between the writes can still leave them at different lengths, so the
    shortest one wins and the ragged tail is discarded and regenerated.
    """
    if not all(os.path.exists(p) for p in (raw_path, eval_path, counts_path)):
        return [], [], []
    with open(raw_path, encoding='utf-8') as f:
        raws = [l.rstrip('\n') for l in f if l.strip()]
    with open(eval_path, encoding='utf-8') as f:
        evals = [l.rstrip('\n') for l in f if l.strip()]
    with open(counts_path) as f:
        counts = json.load(f).get('raw_verse_counts', [])
    k = min(len(raws), len(evals), len(counts))
    return raws[:k], evals[:k], counts[:k]


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()

    outdir = os.path.join(ROOT, 'results', 'dream_baseline')
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, 'dream7b_raw.txt')
    eval_path = os.path.join(outdir, 'dream7b_sonnets.txt')
    counts_path = os.path.join(outdir, 'dream7b_raw_verse_counts.json')

    raws, evals, raw_counts = load_resume(raw_path, eval_path, counts_path)
    if raws:
        print(f"resuming -- {len(raws)}/{a.n} sonnets already generated")
    if len(raws) >= a.n:
        print("nothing left to generate")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to('cuda').eval()

    for i in range(len(raws), a.n):
        tema = TEMAS[i % len(TEMAS)]
        messages = [{'role': 'user', 'content': PROMPT.format(tema=tema)}]
        enc = tok.apply_chat_template(messages, return_tensors='pt',
                                      return_dict=True, add_generation_prompt=True)
        ids = enc['input_ids'].to('cuda')
        attn = enc['attention_mask'].to('cuda')
        # These are the values from the official example in HKUNLP/Dream's
        # README, not a guess. Two earlier attempts to "fix" degenerate output
        # by changing them made things worse, because the decoding parameters
        # were never the problem.
        out = model.diffusion_generate(
            ids, attention_mask=attn, max_new_tokens=512,
            output_history=False, return_dict_in_generate=True,
            steps=512, temperature=0.2, top_p=0.95,
            alg='entropy', alg_temp=0.0)

        # A diffusion LM fills the whole max_new_tokens canvas: the answer ends
        # at the first eos and everything after it is filler. The official
        # example decodes WITH special tokens and splits on eos for exactly
        # this reason. Passing skip_special_tokens=True instead removes the eos
        # marker while keeping the filler, which is what produced the earlier
        # "degenerate" output -- sonnets of 32 and 66 verses, and one that was
        # the word "ha" repeated fifty times. That was our post-processing, not
        # the model.
        text = tok.decode(out.sequences[0][ids.shape[1]:])
        text = text.split(tok.eos_token)[0]
        line, n_raw = to_eval_line(text)
        raws.append(text.replace('\n', ' || '))
        evals.append(line)
        raw_counts.append(n_raw)
        print(f"[{i+1}/{a.n}] {tema}: {n_raw} versos crudos "
              f"({'formato ok' if n_raw == 14 else 'formato FALLA'})", flush=True)
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(raws) + '\n')
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(evals) + '\n')
        with open(counts_path, 'w') as f:
            json.dump(dict(raw_verse_counts=raw_counts,
                           format_pct_measured=100.0 * sum(
                               1 for c in raw_counts if c == 14) / len(raw_counts)),
                      f, indent=1)

    exact = sum(1 for c in raw_counts if c == 14)
    print(f"\nformato medido (14 versos exactos antes de truncar): "
          f"{exact}/{len(raw_counts)} = {100.0 * exact / len(raw_counts):.1f}%")
    print("Guardado:", eval_path)


if __name__ == '__main__':
    main()
