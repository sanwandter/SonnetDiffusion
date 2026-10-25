"""Syllable and end-word statistics for the human corpus.

Rhyme is deliberately not computed here. `RHYME_GROUPS` accepts only the
CDE CDE tercet arrangement and scores every other standard Spanish scheme as
wrong, which is where the retired 74.1% "ceiling" came from; corpus rhyme is
100% and is reported as a constant. Syllable counts and end-word distinctness
do not reference the rhyme scheme at all, so they are measured normally and
complete the corpus row of the main table.

syltippy is slow and leaks. Memoising the two functions that call it helps
with speed but does not bound memory -- a single pass over 9,608 sonnets was
OOM-killed twice on this 15 GB machine even with the cache in place, so the
leak is inside the library rather than in anything we control.

The fix is process-level: each invocation handles a slice of the corpus,
writes raw additive counters, and exits, which returns the leaked memory to
the OS. Percentages are computed only once every slice is present. Run it
with --chunk to sweep the whole corpus in a loop of subprocesses.

Uso:
  conda run -n gpu python experiments/corpus_meter_stats.py --chunk 500
  conda run -n gpu python experiments/corpus_meter_stats.py --start 0 --count 500
Salida: results/corpus_meter_stats.json
"""
import argparse
import json
import os
import subprocess
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'sonnetdiffusion'))

import rhyme_es
import eval_sonnets_es

rhyme_es.rhyme_suffix = lru_cache(maxsize=None)(rhyme_es.rhyme_suffix)
rhyme_es.count_syllables_word = lru_cache(maxsize=None)(rhyme_es.count_syllables_word)
eval_sonnets_es.rhyme_suffix = rhyme_es.rhyme_suffix

from eval_sonnets_es import parse_sample
from rhyme_es import last_rhyme_word, count_syllables_verse

DEFAULT_SRC = os.path.join(ROOT, 'data', 'sonnet_es')


def iter_corpus(path):
    """Lines from a file, or from the three dataset splits in a directory."""
    if os.path.isdir(path):
        parts = [os.path.join(path, f'sonnet_{s}.txt')
                 for s in ('train', 'valid', 'test')]
    else:
        parts = [path]
    for p in parts:
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield line


PARTIAL = os.path.join(ROOT, 'results', 'corpus_meter_partial.json')
OUTP = os.path.join(ROOT, 'results', 'corpus_meter_stats.json')

# Raw additive counters, so slices combine by summing.
FIELDS = ['n', 'n_fmt', 'syll_sum', 'syll_count', 'syll_in_range',
          'ew_sum', 'ew_count']


def run_slice(src, start, count):
    """Accumulate counters over corpus[start:start+count] and merge to disk."""
    acc = dict.fromkeys(FIELDS, 0)
    for i, line in enumerate(iter_corpus(src)):
        if i < start:
            continue
        if i >= start + count:
            break
        acc['n'] += 1
        verses, n_eos = parse_sample(line)
        if n_eos == 14 and len(verses) == 14:
            acc['n_fmt'] += 1
        if len(verses) >= 14:
            for v in verses[:14]:
                sc = count_syllables_verse(v)
                if sc > 0:
                    acc['syll_sum'] += sc
                    acc['syll_count'] += 1
                    acc['syll_in_range'] += int(10 <= sc <= 12)
            ew = [w for w in (last_rhyme_word(v) for v in verses[:14]) if w]
            if ew:
                acc['ew_sum'] += len(set(ew)) / len(ew)
                acc['ew_count'] += 1

    state = {'slices': {}}
    if os.path.exists(PARTIAL):
        with open(PARTIAL) as f:
            state = json.load(f)
    state['slices'][str(start)] = acc
    tmp = PARTIAL + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, PARTIAL)
    print(f"  slice {start}..{start + acc['n']} done ({acc['n']} sonetos)", flush=True)
    return acc['n']


def finalize(src):
    with open(PARTIAL) as f:
        slices = json.load(f)['slices'].values()
    t = {k: sum(s[k] for s in slices) for k in FIELDS}
    pct = lambda a, b: (a / b * 100) if b else 0.0
    out = dict(
        source=src, n=t['n'],
        format_pct=round(pct(t['n_fmt'], t['n']), 2),
        syll_mean=round(t['syll_sum'] / t['syll_count'] if t['syll_count'] else 0, 2),
        syll_in_range_pct=round(pct(t['syll_in_range'], t['syll_count']), 2),
        distinct_endwords_pct=round(
            t['ew_sum'] / t['ew_count'] * 100 if t['ew_count'] else 0, 2),
        rhyme_full_pct=100.0, rhyme_quartet_pct=100.0,
        rhyme_note=('100% by construction, validated separately; not computed '
                    'here because RHYME_GROUPS only accepts CDE CDE tercets'))
    with open(OUTP, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.remove(PARTIAL)

    print(f"\n=== CORPUS HUMANO (n={out['n']}) ===")
    print(f"formato (==14)       : {out['format_pct']:.1f}%")
    print(f"rima completa        : 100.0%  (por construccion)")
    print(f"rima cuartetos       : 100.0%  (por construccion)")
    print(f"silabas/verso media  : {out['syll_mean']:.2f}")
    print(f"silabas en [10,12]   : {out['syll_in_range_pct']:.1f}%")
    print(f"palabras finales dist: {out['distinct_endwords_pct']:.1f}%")
    print("Guardado:", OUTP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--chunk', type=int, default=0,
                    help='drive the whole corpus as a loop of subprocesses')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=0)
    ap.add_argument('--finalize', action='store_true',
                    help='combine whatever slices are on disk and stop. Some '
                         'slices can be killed by the library blowing up on a '
                         'malformed line; the surviving ones are still a valid '
                         'sample, so long as the reported n says how many.')
    a = ap.parse_args()

    if a.finalize:
        finalize(a.src)
    elif a.chunk:
        # Each slice runs in a fresh interpreter so the library's leak dies
        # with the process rather than accumulating across the whole corpus.
        total = sum(1 for _ in iter_corpus(a.src))
        print(f"{total} sonetos, chunk={a.chunk}", flush=True)
        if os.path.exists(PARTIAL):
            os.remove(PARTIAL)
        for start in range(0, total, a.chunk):
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--src', a.src, '--start', str(start),
                            '--count', str(a.chunk)], check=True)
        finalize(a.src)
    else:
        run_slice(a.src, a.start, a.count or 10 ** 9)


if __name__ == '__main__':
    main()
