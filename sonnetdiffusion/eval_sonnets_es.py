"""Evaluación de sonetos españoles: formato (14 versos), precisión de rima
y conteo aproximado de sílabas.

Uso:
  conda run -n gpu python sonnetdiffusion/eval_sonnets_es.py <muestras.txt>
  conda run -n gpu python sonnetdiffusion/eval_sonnets_es.py train
  conda run -n gpu python sonnetdiffusion/eval_sonnets_es.py <ckpt.pt> --sample 8
"""
import argparse
import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rhyme_es import (RHYME_GROUPS, QUARTET_GROUPS, last_rhyme_word,
                      rhyme_suffix, count_syllables_verse)

EOS_RE = re.compile(r"<\s*eos\s*>")
STOP = {"END", "PAD"}
SKIP = {"START", "UNK"}


def parse_sample(line):
    """Parsea una línea de muestra a (versos, n_eos)."""
    toks = line.split()
    content = []
    for t in toks:
        if t in STOP:
            break
        if t in SKIP:
            continue
        content.append(t)
    s = " ".join(content)
    n_eos = len(EOS_RE.findall(s))
    parts = EOS_RE.split(s)
    verses = [p.split() for p in parts if p.split()]
    return verses, n_eos


def group_rhyme_counts(verses, groups):
    last = {}
    for i, v in enumerate(verses[:14], start=1):
        w = last_rhyme_word(v)
        last[i] = rhyme_suffix(w) if w else None
    good = tot = 0
    for lines in groups.values():
        sufs = [last[l] for l in lines if last.get(l)]
        for a in range(len(sufs)):
            for b in range(a + 1, len(sufs)):
                tot += 1
                good += int(sufs[a] == sufs[b])
    return good, tot


def compute_metrics(lines, show=0):
    """Métricas de formato, rima y sílabas en UNA pasada (streaming).

    `lines` puede ser un iterador: no se cargan todas las muestras en memoria y
    los acumuladores son escalares -> memoria O(1) sin importar el tamaño del
    corpus (evita el OOM al evaluar los 8648 sonetos de `train`). Si `show > 0`
    recolecta las primeras muestras como ejemplos en la clave "_examples".
    """
    n = n_fmt = 0
    g_all = t_all = g_qua = t_qua = perfect = 0
    syll_sum = syll_count = syll_in_range = 0
    ew_rate_sum = ew_rate_count = 0
    examples = []
    for line in lines:
        n += 1
        verses, n_eos = parse_sample(line)
        if show and len(examples) < show and len(verses) >= 4:
            examples.append(verses[:14])
        is_fmt = (n_eos == 14 and len(verses) == 14)
        n_fmt += int(is_fmt)
        if len(verses) >= 14:
            ga, ta = group_rhyme_counts(verses, RHYME_GROUPS)
            gq, tq = group_rhyme_counts(verses, QUARTET_GROUPS)
            g_all += ga; t_all += ta
            g_qua += gq; t_qua += tq
            if ta > 0 and ga == ta:
                perfect += 1
            for v in verses[:14]:
                sc = count_syllables_verse(v)
                if sc > 0:
                    syll_sum += sc
                    syll_count += 1
                    syll_in_range += int(10 <= sc <= 12)
            ew = [w for w in (last_rhyme_word(v) for v in verses[:14]) if w]
            if ew:
                ew_rate_sum += len(set(ew)) / len(ew)
                ew_rate_count += 1
    return {
        "n": n,
        "format_pct": n_fmt / n * 100 if n else 0,
        "rhyme_full_pct": g_all / t_all * 100 if t_all else 0,
        "rhyme_quartet_pct": g_qua / t_qua * 100 if t_qua else 0,
        "perfect_scheme": perfect,
        "syll_mean": syll_sum / syll_count if syll_count else 0,
        "syll_in_range_pct": syll_in_range / syll_count * 100 if syll_count else 0,
        "distinct_endwords_pct": (ew_rate_sum / ew_rate_count * 100
                                  if ew_rate_count else 0),
        "_examples": examples,
    }


def evaluate(lines, show=4):
    m = compute_metrics(lines, show=show)
    n = m["n"]
    print(f"n muestras           : {n}")
    print(f"formato acc (==14)   : {m['format_pct']:.1f}%")
    if m["rhyme_full_pct"] > 0 or n > 0:
        print(f"rima pares acc (ABBA ABBA CDE CDE): {m['rhyme_full_pct']:.1f}%")
    if m["rhyme_quartet_pct"] > 0 or n > 0:
        print(f"rima pares acc (cuartetos ABBA ABBA): {m['rhyme_quartet_pct']:.1f}%")
    print(f"sonetos esquema perfecto: {m['perfect_scheme']}")
    if m["syll_mean"] > 0:
        print(f"sílabas/verso (aprox): media {m['syll_mean']:.1f}, en[10,12] {m['syll_in_range_pct']:.1f}%")
    if m["distinct_endwords_pct"] > 0:
        print(f"palabras finales distintas/verso: {m['distinct_endwords_pct']:.1f}%  (100% = sin repetición)")
    print("\n--- ejemplos ---")
    for ex in m["_examples"]:
        print("·" * 50)
        for v in ex[:14]:
            print("   " + " ".join(v))
    print("·" * 50)


def load_lines(arg):
    root = os.path.dirname(HERE)
    corpus = {k: os.path.join(root, "data", "sonnet_es", f"sonnet_{k}.txt")
              for k in ("train", "valid", "test")}
    path = corpus[arg] if arg in corpus else arg
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="archivo .txt de muestras, o train/valid/test")
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="evaluar solo primeras N líneas")
    args = ap.parse_args()
    lines = load_lines(args.input)
    if args.limit:
        lines = itertools.islice(lines, args.limit)
    print(f"== eval: {args.input} ==")
    evaluate(lines, show=args.show)
