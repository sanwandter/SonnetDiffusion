"""Convierte corpora de sonetos españoles (DISCO + Siglo de Oro) al formato
e2e-tgt de PoetryDiffusion.

Salida (en datasets/sonnet_es/):
  sonnet_train.txt / sonnet_valid.txt / sonnet_test.txt
    Un soneto por línea, 14 versos cada uno, cada verso seguido de ' <eos> ',
    en minúsculas y tokenizado con spaCy Spanish.

Ejecutar:  conda run -n gpu python datasets/prepare_sonnets_es.py
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
OUT_DIR = os.path.join(HERE, "sonnet_es")

SOURCES = [
    os.path.join(ROOT, "disco_sonnets.jsonl"),
    os.path.join(ROOT, "golden_age_sonnets.jsonl"),
]

SEED = 102
VALID_FRAC = 0.05
TEST_FRAC = 0.05
VOCAB_MIN_FREQ = 10


def load_sonnets():
    """Genera texto crudo de sonetos desde las fuentes jsonl."""
    for path in SOURCES:
        if not os.path.exists(path):
            print(f"  AVISO: falta {path}")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                txt = obj.get("text", "")
                if txt:
                    yield txt
                    n += 1
        print(f"  leídos {n} sonetos de {os.path.basename(path)}")


def build():
    import spacy
    from spacy.lang.es import Spanish

    tokenizer = Spanish().tokenizer

    kept = []
    n_total = 0
    n_bad_lines = 0
    for txt in load_sonnets():
        n_total += 1
        verses = [v.strip() for v in txt.split("\n") if v.strip()]
        if len(verses) != 14:
            n_bad_lines += 1
            continue
        out_verses = []
        ok = True
        for v in verses:
            toks = [t.text for t in tokenizer(v.lower()) if t.text.strip()]
            if not toks:
                ok = False
                break
            out_verses.append(" ".join(toks))
        if not ok:
            continue
        line = " ".join(v + " <eos>" for v in out_verses)
        kept.append(line)

    print(f"\n  total sonetos leídos  : {n_total}")
    print(f"  descartados (!=14 vrs): {n_bad_lines}")
    print(f"  conservados (14 vrs)  : {len(kept)}")

    # deduplicar sonetos idénticos (corpora se solapan)
    before = len(kept)
    kept = list(dict.fromkeys(kept))
    print(f"  deduplicados          : {before - len(kept)} eliminados -> {len(kept)}")

    # estadísticas de largo (tokens por línea)
    lengths = [len(l.split(" ")) for l in kept]
    lengths.sort()
    n = len(lengths)
    mean = sum(lengths) / n
    p50 = lengths[n // 2]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[int(n * 0.99)]
    mx = lengths[-1]
    print("\n  tokens por soneto (incl. <eos>):")
    print(f"    media={mean:.1f}  p50={p50}  p95={p95}  p99={p99}  max={mx}")
    for sz in (12, 14, 16, 18, 20):
        cap = sz * sz
        frac = sum(1 for L in lengths if L > cap) / n
        print(f"    image_size={sz} (cap {cap}): {frac*100:.2f}% sonetos exceden (truncados)")

    # mezclar + dividir
    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_valid = int(len(kept) * VALID_FRAC)
    n_test = int(len(kept) * TEST_FRAC)
    valid = kept[:n_valid]
    test = kept[n_valid:n_valid + n_test]
    train = kept[n_valid + n_test:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        p = os.path.join(OUT_DIR, f"sonnet_{name}.txt")
        with open(p, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")
        print(f"  escritos {len(data):>6} -> {p}")

    # estimación de tamaño de vocabulario (desde TRAIN, freq > VOCAB_MIN_FREQ)
    counter = Counter()
    for line in train:
        counter.update(line.split(" "))
    kept_vocab = [w for w, c in counter.items() if c > VOCAB_MIN_FREQ]
    vocab_size = len(kept_vocab) + 4
    print(f"\n  tokens únicos en train       : {len(counter)}")
    print(f"  vocab conservado (freq>{VOCAB_MIN_FREQ}) + 4 : {vocab_size}")
    print(f"  -> usar --vocab_size >= {vocab_size} (con margen)")


if __name__ == "__main__":
    build()
