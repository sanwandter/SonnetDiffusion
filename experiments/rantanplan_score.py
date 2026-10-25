"""Independent rhyme and meter analysis with Rantanplan (review response).

Rantanplan (LINHD/POSTDATA, UNED) is a Spanish scansion tool with its own
syllabifier, POS-based stress assignment, synalepha/synaeresis handling and
rhyme-scheme detection. None of that is shared with our suffix rule or with
syltippy, so it scores rhyme and meter with a representation the controller
never optimized. (It shares the textbook DEFINITION of consonant rhyme --
identity from the last stressed vowel -- as any correct measure must.)

Rhyme is recorded two ways. `rhyme`/`rhyme_type` are Rantanplan's own scheme
letters, which come out of a greedy grouping of the whole poem (a line can be
labelled assonant because of a different line). `cons`/`asso` are per-line
codes taken one step earlier, from Rantanplan's own stressed endings and its
own cleaning rules (get_stressed_endings + get_clean_codes), before any
grouping: two lines rhyme consonantly iff their `cons` codes are equal. This
is the measure strict_metrics.py reports; it depends on no other line.

It needs spaCy 2.2, so it runs in its own env and cannot import the rest of
the codebase. strict_metrics.py dumps every sonnet to score; this script
writes one JSON record per sonnet to a cache keyed by the sonnet's sha1;
strict_metrics.py reads the cache back.

Setup (once):
  conda create -n rantanplan python=3.8
  conda activate rantanplan
  pip install "spacy>=3.0,<3.2" spacy-affixes      # setup.py imports spacy
  pip install --no-build-isolation rantanplan        # pins spacy 2.2.4
  pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_md-2.2.5/es_core_news_md-2.2.5.tar.gz

Uso:
  conda run -n gpu python experiments/strict_metrics.py --dump-lines results/rantanplan_inputs.txt
  conda run -n rantanplan python experiments/rantanplan_score.py results/rantanplan_inputs.txt
Salida: results/rantanplan_cache_v2.jsonl
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, 'results', 'rantanplan_cache_v2.jsonl')
EOS = re.compile(r'<\s*eos\s*>')


def key(line):
    return hashlib.sha1(line.strip().encode('utf-8')).hexdigest()


def to_poem(line):
    """First 14 verses, one per line, letters only (every system is scored
    without punctuation, since the diffusion outputs carry none)."""
    verses = []
    for v in EOS.split(line):
        v = re.sub(r'[^\w\s]|\d|_', ' ', v)
        v = re.sub(r'\s+', ' ', v).strip()
        if v and v not in ('START', 'END', 'PAD', 'UNK'):
            verses.append(v)
    return verses[:14]


def line_codes(lines, assonance, relaxation):
    """Per-line rhyme ending under Rantanplan's own rules, or None for a line
    whose stress it cannot place. Computed before any grouping."""
    from rantanplan.rhymes import get_stressed_endings, get_clean_codes
    out = []
    for line in lines:
        try:
            se = get_stressed_endings([line])   # mutated below; one per call
            c2e, codes = get_clean_codes(se, assonance, relaxation)
            out.append(c2e[codes[0]].lower())
        except (ValueError, IndexError, KeyError):
            out.append(None)
    return out


def analyse(verses):
    from rantanplan import get_scansion
    # always_return_rhyme: by default Rantanplan assigns rhyme letters only
    # when it recognises the whole poem as a known stanza type, and returns
    # None for every line otherwise -- which turns a rhyme measure into a
    # sonnet classifier (and scored a near-sonnet with clean ABBA rhymes as 0).
    out = get_scansion('\n'.join(verses), rhyme_analysis=True,
                       always_return_rhyme=True)
    cons = line_codes(out, assonance=False, relaxation=True)
    cons_strict = line_codes(out, assonance=False, relaxation=False)
    asso = line_codes(out, assonance=True, relaxation=True)
    return [dict(rhyme=x.get('rhyme'), ending=x.get('ending'),
                 rhyme_type=x.get('rhyme_type'),
                 cons=cons[i], cons_strict=cons_strict[i], asso=asso[i],
                 length=(x.get('rhythm') or {}).get('length'),
                 structure=x.get('structure')) for i, x in enumerate(out)]


def main():
    src = sys.argv[1]
    done = set()
    if os.path.exists(CACHE):
        with open(CACHE, encoding='utf-8') as f:
            done = {json.loads(l)['key'] for l in f if l.strip()}
    with open(src, encoding='utf-8') as f:
        todo = [l for l in f if l.strip() and key(l) not in done]
    print('%d sonnets to score, %d cached' % (len(todo), len(done)), flush=True)
    with open(CACHE, 'a', encoding='utf-8') as out:
        for n, line in enumerate(todo, 1):
            verses = to_poem(line)
            rec = dict(key=key(line), n_verses=len(verses))
            try:
                rec['lines'] = analyse(verses) if verses else []
            except Exception as e:   # one malformed sonnet must not stop the run
                rec['error'] = repr(e)[:200]
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if n % 500 == 0:
                out.flush()
                print('  %d/%d' % (n, len(todo)), flush=True)
    print('done ->', CACHE)


if __name__ == '__main__':
    main()
