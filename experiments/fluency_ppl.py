"""Fluency under independent language models: perplexity, bits per character,
and model-free diversity measures (EvoMUSART resubmission).

Every system is scored on the SAME normalisation, because our diffusion model
emits lowercase text without punctuation and punctuation alone moves
perplexity: verses come from eval_sonnets_es.parse_sample (the evaluator's own
split), are lowercased, keep only letters, and are joined one verse per line.
Human sonnets, Haiku and Dream go through exactly this path.

Per poem and per scoring model:
  ppl  exp(mean token NLL) under the model's own tokenizer, from BOS, no prompt
  bpc  total NLL in bits / number of characters (tokenizer-independent, so it
       can be compared across the two scoring models)
Model-free, per poem: word unigram entropy (bits), distinct-1, distinct-2 and
repetition rate (1 - distinct-1).

Scoring models (base models, near full precision; see
paper/evomusart/EVOMUSART_2027.md):
  salamandra   BSC-LT/salamandra-7b   bf16
  gemma4       google/gemma-4-12B     8-bit (bf16 does not fit in 24 GB); loaded
               from a copy split into 4 GB shards, because WSL (15 GB RAM)
               refuses to mmap the hub's single 24 GB safetensors file
A ranking of systems counts as a finding only if both models agree.

Contamination: canonical Spanish sonnets are in these models' training data,
so the human row can be low because it is memorised. It is a reference, not a
target.

Uso:
  conda run -n gpu python experiments/fluency_ppl.py --model salamandra
  conda run -n gpu python experiments/fluency_ppl.py --model gemma4
  conda run -n gpu python experiments/fluency_ppl.py --report
Salida: results/fluency/cache_<model>.jsonl, results/fluency/fluency.json (+ table)
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import strict_metrics as sm
import harness
from eval_sonnets_es import parse_sample

OUT = os.path.join(ROOT, 'results', 'fluency')
MODELS = {
    'salamandra': dict(repo='BSC-LT/salamandra-7b', quant=None),
    'gemma4': dict(repo=os.path.expanduser('~/.cache/huggingface/resharded/gemma-4-12B'), quant='8bit'),
}
SYSTEMS = ['human', 'human_clean', 'haiku_single', 'haiku', 'haiku_v2', 'dream', 'nocontrol', 'bon6', 'bon168', 'K2', 'K5',
           'K5_repel', 'K5_cdcdcd']
LETTERS = re.compile(r"[^a-záéíóúüñ]+")


def normalise(line):
    """Verses as the evaluator sees them, lowercase, letters only, one per line."""
    verses, _ = parse_sample(line)
    out = []
    for v in verses[:14]:
        words = [w for w in (LETTERS.sub('', t.lower()) for t in v) if w]
        if words and words not in (['eos'],):
            out.append(' '.join(words))
    return '\n'.join(out)


def key(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def diversity(text):
    words = text.split()
    if not words:
        return dict(entropy=0.0, distinct1=0.0, distinct2=0.0, repetition=1.0, n_words=0)
    c = Counter(words)
    n = len(words)
    ent = -sum(v / n * math.log2(v / n) for v in c.values())
    bigrams = list(zip(words, words[1:]))
    d1 = len(c) / n
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return dict(entropy=ent, distinct1=d1, distinct2=d2, repetition=1 - d1, n_words=n)


def load_scorer(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[name]
    tok = AutoTokenizer.from_pretrained(spec['repo'])
    kw = dict(dtype=torch.bfloat16, device_map='cuda')
    if spec['quant'] == '8bit':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(spec['repo'], **kw).eval()
    return tok, model


def nll(tok, model, text):
    """(sum of token NLL in nats, number of scored tokens) for text after BOS."""
    import torch
    ids = tok(text, return_tensors='pt', add_special_tokens=True).input_ids
    if ids[0, 0].item() != tok.bos_token_id and tok.bos_token_id is not None:
        ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1)
    ids = ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    tok_nll = -lp.gather(1, tgt[:, None]).squeeze(1)
    return tok_nll.sum().item(), tgt.numel()


def systems_texts():
    phon = sm.make_phon_key()
    systems = sm.load_systems(SYSTEMS, phon)
    return {name: [[normalise(l) for l in rep] for rep in reps]
            for name, reps in systems.items()}


def score(model_name):
    os.makedirs(OUT, exist_ok=True)
    cache_path = os.path.join(OUT, f'cache_{model_name}.jsonl')
    done = set()
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            done = {json.loads(l)['key'] for l in f if l.strip()}
    texts = {t for reps in systems_texts().values() for rep in reps for t in rep if t}
    todo = [t for t in sorted(texts) if key(t) not in done]
    print(f"[{model_name}] {len(todo)} poems to score, {len(done)} cached", flush=True)
    if not todo:
        return
    tok, model = load_scorer(model_name)
    with open(cache_path, 'a', encoding='utf-8') as f:
        for i, t in enumerate(todo, 1):
            s, n = nll(tok, model, t)
            f.write(json.dumps(dict(key=key(t), nll=s, n_tokens=n,
                                    n_chars=len(t))) + '\n')
            if i % 100 == 0:
                f.flush()
                print(f"  {i}/{len(todo)}", flush=True)


def report():
    caches = {}
    for m in MODELS:
        p = os.path.join(OUT, f'cache_{m}.jsonl')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                caches[m] = {r['key']: r for r in (json.loads(l) for l in f if l.strip())}
    results = {}
    for name, reps in systems_texts().items():
        per_rep = []
        for rep in reps:
            rep = [t for t in rep if t]
            out = {}
            for k in ('entropy', 'distinct1', 'distinct2', 'repetition'):
                out[k] = sum(diversity(t)[k] for t in rep) / len(rep)
            for m, cache in caches.items():
                rows = [cache[key(t)] for t in rep if key(t) in cache]
                if len(rows) != len(rep):
                    continue
                # geometric mean of per-poem perplexity = exp(mean per-poem log ppl)
                out[f'ppl_{m}'] = math.exp(sum(r['nll'] / r['n_tokens'] for r in rows) / len(rows))
                out[f'bpc_{m}'] = sum(r['nll'] / math.log(2) / r['n_chars'] for r in rows) / len(rows)
            per_rep.append(out)
        keys = [k for k in per_rep[0] if all(k in r for r in per_rep)]
        results[name] = dict(replicates=len(per_rep),
                             aggregate=harness.aggregate_seeds(per_rep, keys),
                             per_replicate=per_rep)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'fluency.json'), 'w') as f:
        json.dump(results, f, indent=1)
    cols = [k for k in ('ppl_salamandra', 'bpc_salamandra', 'ppl_gemma4', 'bpc_gemma4',
                        'entropy', 'distinct1', 'distinct2', 'repetition')
            if any(k in r['aggregate'] for r in results.values())]
    print('| System | ' + ' | '.join(cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for name, r in results.items():
        cells = []
        for k in cols:
            m = r['aggregate'].get(k)
            cells.append('—' if m is None else (f"{m['mean']:.2f}" if m['halfwidth'] == 0
                                                else f"{m['mean']:.2f} ± {m['halfwidth']:.2f}"))
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    report_adjusted(caches)


def report_adjusted(caches, base='nocontrol', others=('bon168', 'K5', 'K5_cdcdcd')):
    """Per-poem OLS, log ppl ~ 1 + repetition + [system]: how much of each
    system's perplexity drop against `base` survives holding repetition fixed.
    Poems are treated as independent (90 per system), not per-seed."""
    import numpy as np
    texts = systems_texts()
    print('\n| Scorer | System vs ' + base + ' | raw Δ log ppl | Δ adjusted for repetition (95% CI) | repetition slope |')
    print('|---|---|---|---|---|')
    for m, cache in caches.items():
        for s in others:
            if s not in texts or base not in texts:
                continue
            X, y = [], []
            for name in (base, s):
                for t in (t for rep in texts[name] for t in rep if t):
                    r = cache[key(t)]
                    X.append([1.0, diversity(t)['repetition'], float(name == s)])
                    y.append(r['nll'] / r['n_tokens'])
            X, y = np.array(X), np.array(y)
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            se = np.sqrt(res @ res / (len(y) - 3) * np.linalg.inv(X.T @ X).diagonal())
            raw = y[X[:, 2] == 1].mean() - y[X[:, 2] == 0].mean()
            print(f'| {m} | {s} | {raw:+.3f} | {b[2]:+.3f} ± {1.96 * se[2]:.3f} | {b[1]:+.2f} |')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS))
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--systems', nargs='*', default=None)
    a = ap.parse_args()
    if a.systems:
        SYSTEMS = a.systems
    if a.model:
        score(a.model)
    if a.report:
        report()
