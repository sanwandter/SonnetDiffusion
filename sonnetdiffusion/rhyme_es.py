"""Utilidades de rima fonética y conteo de sílabas en español.

Clave de rima = desde el núcleo vocálico acentuado hasta el final de la palabra,
acentos normalizados, glide de diptongo ascendente excluido.

Usado por:
  - eval_sonnets_es.py (métricas de formato, rima y sílabas)
  - controllers_es.py (build_suffix_matrix para el controlador analítico de rima)
"""
import unicodedata
from syltippy import syllabize

VOWELS = set("aeiouáéíóúü")
ACCENTED = set("áéíóú")
OPEN_VOWELS = set("aeo")
WEAK_VOWELS = set("iuü")

# Tokens especiales de PoetryDiffusion (no llevan rima)
SPECIAL = {"START", "END", "UNK", "PAD", "<", "eos", ">"}

# Esquema canónico del soneto español ABBA ABBA CDE CDE (versos 1-indexed)
RHYME_GROUPS = {"A": [1, 4, 5, 8], "B": [2, 3, 6, 7],
                "C": [9, 12], "D": [10, 13], "E": [11, 14]}
# Solo cuartetos, para evaluar la octava ABBA ABBA
QUARTET_GROUPS = {"A": [1, 4, 5, 8], "B": [2, 3, 6, 7]}


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _nucleus_index(syllable: str):
    """Índice de la vocal que lleva el peso prosódico (núcleo silábico)."""
    for i, ch in enumerate(syllable):
        if ch in ACCENTED:
            return i
    for i, ch in enumerate(syllable):
        if ch in OPEN_VOWELS:
            return i
    last = None
    for i, ch in enumerate(syllable):
        if ch in WEAK_VOWELS:
            last = i
    return last


def rhyme_suffix(word: str):
    """Clave de rima consonante en español, o None si no rima."""
    word = word.lower().strip()
    if not word or not any(c in VOWELS for c in word):
        return None
    try:
        sylls, stress = syllabize(word)
    except Exception:
        return None
    if not sylls:
        return None
    stress = max(0, min(stress, len(sylls) - 1))
    stressed = sylls[stress]
    rest = "".join(sylls[stress + 1:])
    nidx = _nucleus_index(stressed)
    if nidx is None:
        tail = stressed + rest
        for i, ch in enumerate(tail):
            if ch in VOWELS:
                return strip_accents(tail[i:])
        return None
    return strip_accents(stressed[nidx:] + rest)


def count_syllables_word(word: str) -> int:
    word = word.lower().strip()
    if not word or not any(c in VOWELS for c in word):
        return 0
    try:
        sylls, _ = syllabize(word)
        return len(sylls)
    except Exception:
        return 0


def count_syllables_verse(words) -> int:
    """Conteo aproximado de sílabas métricas con sinalefa heurística."""
    toks = [w for w in words if w not in SPECIAL and any(c.isalpha() for c in w)]
    total = 0
    prev_end_vowel = False
    for w in toks:
        n = count_syllables_word(w)
        if n == 0:
            continue
        starts_vowel = w[0].lower() in VOWELS
        if prev_end_vowel and starts_vowel and total > 0:
            total += n - 1
        else:
            total += n
        prev_end_vowel = w[-1].lower() in VOWELS
    return total


def last_rhyme_word(words):
    """Último token del verso que lleva rima (salta puntuación y eos)."""
    for w in reversed(words):
        if w in SPECIAL:
            continue
        if rhyme_suffix(w) is not None:
            return w
    return None


def build_suffix_matrix(id2word, vocab_size=None):
    """Matriz one-hot S[V, num_sufijos] para el controlador de rima.
    torch se importa lazy para que las funciones de rima/sílabas
    se puedan usar sin cargar torch (RAM limitada durante entrenamiento)."""
    import torch
    vocab_size = max(vocab_size or 0, max(id2word) + 1)
    suffix2idx = {}
    rows = []
    for tid, word in id2word.items():
        if word in SPECIAL:
            continue
        suf = rhyme_suffix(word)
        if suf is None:
            continue
        if suf not in suffix2idx:
            suffix2idx[suf] = len(suffix2idx)
        rows.append((tid, suffix2idx[suf]))
    S = torch.zeros(vocab_size, max(1, len(suffix2idx)))
    for tid, sidx in rows:
        S[tid, sidx] = 1.0
    return S, suffix2idx
