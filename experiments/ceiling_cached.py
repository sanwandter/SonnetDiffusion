"""Techo del corpus, version rapida y sin fuga: memoiza syltippy por palabra.

syltippy.syllabize es lento y fuga memoria (~1GB/500 sonetos). Pero solo hay
decenas de miles de palabras UNICAS en el corpus, asi que cacheamos las dos
funciones que llaman a syltippy (rhyme_suffix, count_syllables_word) con
lru_cache: cada palabra se silabiza una sola vez -> ~16x menos llamadas, memoria
acotada (~el cache de palabras unicas) y rapido. Una sola pasada, contadores
crudos exactos.

Uso: conda run -n gpu python resultados/experimentos/ceiling_cached.py /tmp/all_sonnets_es.txt
"""
import sys, os
from functools import lru_cache
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'vision2sonnet'))

import rhyme_es
import eval_sonnets_es

# Memoiza las dos funciones que invocan a syltippy. Las demas funciones
# (last_rhyme_word, count_syllables_verse, group_rhyme_counts) leen estos
# nombres como globales de modulo en tiempo de llamada, asi que el parche se
# propaga.
rhyme_es.rhyme_suffix = lru_cache(maxsize=None)(rhyme_es.rhyme_suffix)
rhyme_es.count_syllables_word = lru_cache(maxsize=None)(rhyme_es.count_syllables_word)
eval_sonnets_es.rhyme_suffix = rhyme_es.rhyme_suffix

from eval_sonnets_es import parse_sample, group_rhyme_counts
from rhyme_es import RHYME_GROUPS, QUARTET_GROUPS, last_rhyme_word, count_syllables_verse


def main():
    path = sys.argv[1]
    n = n_fmt = 0
    g_all = t_all = g_qua = t_qua = perfect = 0
    syll_sum = syll_count = syll_in_range = 0
    ew_sum = ew_count = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            verses, n_eos = parse_sample(line)
            if n_eos == 14 and len(verses) == 14:
                n_fmt += 1
            if len(verses) >= 14:
                ga, ta = group_rhyme_counts(verses, RHYME_GROUPS)
                gq, tq = group_rhyme_counts(verses, QUARTET_GROUPS)
                g_all += ga; t_all += ta; g_qua += gq; t_qua += tq
                if ta > 0 and ga == ta:
                    perfect += 1
                for v in verses[:14]:
                    sc = count_syllables_verse(v)
                    if sc > 0:
                        syll_sum += sc; syll_count += 1
                        syll_in_range += int(10 <= sc <= 12)
                ew = [w for w in (last_rhyme_word(v) for v in verses[:14]) if w]
                if ew:
                    ew_sum += len(set(ew)) / len(ew); ew_count += 1
            if n % 2000 == 0:
                print(f"  ...{n} sonetos", flush=True)

    pct = lambda a, b: (a / b * 100) if b else 0.0
    print("\n=== TECHO DEL CORPUS (sonetos reales, 9608) ===")
    print(f"n sonetos            : {n}")
    print(f"formato (==14)       : {pct(n_fmt, n):.1f}%")
    print(f"rima pares (completa): {pct(g_all, t_all):.1f}%")
    print(f"rima pares (cuartet) : {pct(g_qua, t_qua):.1f}%")
    print(f"esquema perfecto     : {perfect} ({pct(perfect, n):.1f}%)")
    print(f"silabas/verso media  : {(syll_sum / syll_count if syll_count else 0):.2f}")
    print(f"silabas en [10,12]   : {pct(syll_in_range, syll_count):.1f}%")
    print(f"palabras finales dist: {(ew_sum / ew_count * 100 if ew_count else 0):.1f}%")


if __name__ == '__main__':
    main()
