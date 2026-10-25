"""Stricter, partly independent scoring of saved sonnets (review response).

Addresses three review points the original evaluator cannot:

  * the rhyme metric shares the controller's suffix representation
    -> rhyme is also scored under a PRONUNCIATION key (b/v, silent h,
       c/qu/k, g/j merged; seseo and yeismo optional), while the controller
       keeps the orthographic rule;
  * repeated words satisfy rhyme, and distinct groups may share one sound
    -> identity-excluded rhyme, and a distinct-groups check;
  * no joint satisfaction
    -> a cumulative pass table: each row adds one constraint to the last.

The fixed scheme ABBA ABBA CDE CDE is what every system was asked for, so it
is the target-aligned measure. Published sonnets vary their tercets (and
occasionally their quatrains), so the human reference is also scored under a
per-poem detected scheme; that option picks the best of several schemes and
therefore flatters random text too, so it is reported for every system and
the uncontrolled model gives its chance level.

Systems are read from results/samples/<config>/seed<k>.txt (regen_samples.py);
best-of-N arms are re-selected offline from the saved pool, both with the
paper's oracle (orthographic full rhyme, as best_of_n.py) and with a strict
oracle that sees the identity-excluded pronunciation score.

Uso:
  conda run -n gpu python experiments/strict_metrics.py
  conda run -n gpu python experiments/strict_metrics.py --systems human haiku dream
Salida: results/strict_metrics.json (+ markdown table on stdout)
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'sonnetdiffusion'))

import rhyme_es
# syltippy is slow; every word is scored many times across systems.
rhyme_es.rhyme_suffix = lru_cache(maxsize=None)(rhyme_es.rhyme_suffix)
rhyme_es.count_syllables_word = lru_cache(maxsize=None)(rhyme_es.count_syllables_word)
import eval_sonnets_es
eval_sonnets_es.rhyme_suffix = rhyme_es.rhyme_suffix
import harness
from rantanplan_score import to_poem   # stdlib-only module; same verse split
from eval_sonnets_es import parse_sample
from rhyme_es import RHYME_GROUPS, rhyme_suffix, last_rhyme_word, count_syllables_verse
from rhyme_es import QUARTET_GROUPS as QUARTET_GROUPS_FIXED

SAMPLES = os.path.join(ROOT, 'results', 'samples')
RR_CACHE = os.path.join(ROOT, 'results', 'rantanplan_cache_v2.jsonl')
RR = {}   # sha1(line) -> Rantanplan record, filled by load_rantanplan()


def load_rantanplan():
    """Independent analysis written by rantanplan_score.py (separate env)."""
    if os.path.exists(RR_CACHE):
        with open(RR_CACHE, encoding='utf-8') as f:
            for l in f:
                if l.strip():
                    rec = json.loads(l)
                    RR[rec['key']] = rec


def rr_key(line):
    import hashlib
    return hashlib.sha1(line.strip().encode('utf-8')).hexdigest()
CORPUS = os.path.join(ROOT, 'data', 'sonnet_es')
HUMAN_N, HUMAN_SEED = 250, 102   # the paper's human reference sample

QUATRAINS = {
    'ABBA ABBA': {'A': [1, 4, 5, 8], 'B': [2, 3, 6, 7]},
    'ABAB ABAB': {'A': [1, 3, 5, 7], 'B': [2, 4, 6, 8]},
}
TERCETS = {
    'CDE CDE': {'C': [9, 12], 'D': [10, 13], 'E': [11, 14]},
    'CDC DCD': {'C': [9, 11, 13], 'D': [10, 12, 14]},
    'CDE DCE': {'C': [9, 13], 'D': [10, 12], 'E': [11, 14]},
    'CDE EDC': {'C': [9, 14], 'D': [10, 13], 'E': [11, 12]},
    'CDE DEC': {'C': [9, 14], 'D': [10, 12], 'E': [11, 13]},
    'CDE CED': {'C': [9, 12], 'D': [10, 14], 'E': [11, 13]},
    'CDC CDC': {'C': [9, 11, 12, 14], 'D': [10, 13]},
}
FIXED = dict(RHYME_GROUPS)
# Target schemes a system can be scored under; `<system>@<scheme>` picks one,
# and a samples directory ending in _<scheme> defaults to it.
TARGETS = {
    'cdecde': FIXED,
    'cdcdcd': {**QUARTET_GROUPS_FIXED, 'C': [9, 11, 13], 'D': [10, 12, 14]},
    'cdedce': {**QUARTET_GROUPS_FIXED, 'C': [9, 13], 'D': [10, 12], 'E': [11, 14]},
}


def split_target(name):
    """'nocontrol@cdcdcd' -> ('nocontrol', TARGETS['cdcdcd']);
    'K5_cdcdcd' -> ('K5_cdcdcd', TARGETS['cdcdcd']); otherwise the paper's."""
    base, _, t = name.partition('@')
    if not t:
        t = next((k for k in TARGETS if base.endswith('_' + k)), 'cdecde')
    if t not in TARGETS:
        sys.exit(f"unknown target scheme '{t}' in {name}; known: {sorted(TARGETS)}")
    return base, TARGETS[t]
QUATR_ONLY = QUATRAINS['ABBA ABBA']
SCHEMES = {f'{q} {t}': {**qg, **tg} for q, qg in QUATRAINS.items()
           for t, tg in TERCETS.items()}   # first entry is the canonical one


# --------------------------------------------------------------------------
# pronunciation-based rhyme key
# --------------------------------------------------------------------------
def make_phon_key(seseo=False, yeismo=False):
    """Rhyme key from the stressed vowel on, by sound rather than spelling.

    Starts from the orthographic suffix (so stress placement is shared with
    the controller's rule) and merges spellings of one sound: b/v, silent h,
    c/qu/k before back vowels, g/j before front vowels, final y as a vowel.
    Distinction (z/c != s) and ll != y are the defaults, matching Castilian
    verse of the corpus; --seseo / --yeismo merge them.
    """
    @lru_cache(maxsize=None)
    def key(word):
        s = rhyme_suffix(word) if word else None
        if s is None:
            return None
        s = s.replace('x', 'ks')            # before X is used as a placeholder
        s = s.replace('ch', 'C')
        s = s.replace('h', '')
        s = s.replace('v', 'b').replace('w', 'b')
        s = re.sub(r'g(?=[ei])', 'X', s)    # g before e/i sounds like j
        s = re.sub(r'gu(?=[ei])', 'g', s)
        s = s.replace('j', 'X')
        s = re.sub(r'qu(?=[ei])', 'k', s)
        s = re.sub(r'c(?=[ei])', 'Z', s)
        s = s.replace('z', 'Z').replace('c', 'k').replace('q', 'k')
        s = s.replace('ll', 'L')
        s = re.sub(r'y$', 'i', s)
        if seseo:
            s = s.replace('Z', 's')
        if yeismo:
            s = s.replace('L', 'y')
        return s
    return key


# --------------------------------------------------------------------------
# per-poem analysis
# --------------------------------------------------------------------------
def pair_counts(ends, groups, keyname, ident_fails):
    """(satisfied, total) within-group pairs; ends[i] = dict(word, orth, phon)."""
    good = tot = 0
    for lines in groups.values():
        xs = [ends[l] for l in lines if ends.get(l) and ends[l][keyname]]
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                tot += 1
                ok = xs[a][keyname] == xs[b][keyname]
                if ident_fails and xs[a]['word'] == xs[b]['word']:
                    ok = False
                good += int(ok)
    return good, tot


def groups_distinct(ends, groups, keyname):
    """No end word of one group shares its sound with an end word of another.

    Defined on every end word rather than on a per-group majority sound: a
    majority vote needs a tie-break when a group's words disagree, and the
    earlier set-based tie-break depended on Python's hash seed, so the metric
    changed from run to run. Every group must have at least one keyed word.
    """
    sets = []
    for lines in groups.values():
        xs = {ends[l][keyname] for l in lines if ends.get(l) and ends[l][keyname]}
        if not xs:
            return False
        sets.append(xs)
    return all(not (sets[i] & sets[j])
               for i in range(len(sets)) for j in range(i + 1, len(sets)))


def analyse(line, phon, target=FIXED):
    """Per-poem analysis. `target` is the rhyme scheme the system was asked
    for: ABBA ABBA CDE CDE for every system in the paper, another tercet
    arrangement for the breadth runs (K5_cdcdcd, K5_cdedce)."""
    verses, n_eos = parse_sample(line)
    r = dict(fmt=(n_eos == 14 and len(verses) == 14), has14=len(verses) >= 14)
    if not r['has14']:
        return r
    ends = {}
    for i, v in enumerate(verses[:14], 1):
        w = last_rhyme_word(v)
        ends[i] = dict(word=w, orth=rhyme_suffix(w), phon=phon(w)) if w else None
    r['pairs'] = {
        'orth': pair_counts(ends, target, 'orth', False),           # = paper metric
        'orth_quatr': pair_counts(ends, QUATR_ONLY, 'orth', False),  # = paper metric
        'phon': pair_counts(ends, target, 'phon', False),
        'phon_noident': pair_counts(ends, target, 'phon', True),
        'phon_noident_quatr': pair_counts(ends, QUATR_ONLY, 'phon', True),
    }
    best = max(SCHEMES, key=lambda s: (lambda g, t: g / t if t else 0)(
        *pair_counts(ends, SCHEMES[s], 'phon', True)))
    r['scheme'] = best
    r['pairs']['phon_noident_detected'] = pair_counts(ends, SCHEMES[best], 'phon', True)
    r['pairs']['phon_detected'] = pair_counts(ends, SCHEMES[best], 'phon', False)
    g, t = r['pairs']['phon']
    r['all_rhyme'] = t > 0 and g == t
    g, t = r['pairs']['phon_detected']
    r['all_rhyme_detected'] = t > 0 and g == t
    words = [e['word'] for e in ends.values() if e]
    r['no_repeat'] = len(words) == 14 and len(set(words)) == 14
    r['distinct'] = groups_distinct(ends, target, 'phon')
    r['distinct_detected'] = groups_distinct(ends, SCHEMES[best], 'phon')
    sylls = [count_syllables_verse(v) for v in verses[:14]]
    r['meter_in_range'] = sum(10 <= s <= 12 for s in sylls)
    r['meter_exact'] = sum(s == 11 for s in sylls)
    rec = RR.get(rr_key(line))
    if rec is not None and len(rec.get('lines') or []) >= 14:
        rl = rec['lines'][:14]
        # Rantanplan's verses (to_poem) and their last words, for identity.
        rwords = [v.split()[-1] if v.split() else None for v in to_poem(line)[:14]]
        cons = asso = letters = tot = 0
        for lines in target.values():
            for a in range(len(lines)):
                for b in range(a + 1, len(lines)):
                    i, j = lines[a] - 1, lines[b] - 1
                    x, y = rl[i], rl[j]
                    tot += 1
                    if rwords[i] is not None and rwords[i] == rwords[j]:
                        continue                  # a repeated word is a miss
                    cons += int(x['cons'] is not None and x['cons'] == y['cons'])
                    asso += int(x['asso'] is not None and x['asso'] == y['asso'])
                    letters += int(x['rhyme'] not in (None, '-')
                                   and x['rhyme'] == y['rhyme']
                                   and x['rhyme_type'] == y['rhyme_type'] == 'consonant')
        r['rr'] = dict(cons=cons, asso=asso, letters=letters, tot=tot,
                       exact=sum(x['length'] == 11 for x in rl),
                       sonnet=rl[0]['structure'] == 'sonnet')
    return r


def rr_exact(r):
    """Verses Rantanplan scans as exactly 11 syllables (0 if not scanned)."""
    return r['rr']['exact'] if 'rr' in r else 0


LADDER = [  # (label, predicate on a poem); each row implies the previous ones
    ('14 verses', lambda r: r['fmt']),
    ('+ every scheme pair rhymes', lambda r: r['all_rhyme']),
    ('+ no repeated end word', lambda r: r['no_repeat']),
    ('+ groups use distinct sounds', lambda r: r['distinct']),
    # Meter from Rantanplan: our syllable counter scans only 72.6% of human
    # verses as 11 syllables, Rantanplan 92.4%, so ours is the noisier rule.
    ('+ >=12/14 verses 11 syll. (Rantanplan)', lambda r: rr_exact(r) >= 12),
    ('+ 14/14 verses 11 syll. (Rantanplan)', lambda r: rr_exact(r) == 14),
]
LADDER_OWN_METER = LADDER[:4] + [   # same ladder, meter from our own counter
    ('+ >=12/14 verses 10-12 syll. (own counter)', lambda r: r['meter_in_range'] >= 12),
    ('+ 14/14 verses 10-12 syll. (own counter)', lambda r: r['meter_in_range'] == 14),
]
LADDER_DETECTED = [
    ('14 verses', lambda r: r['fmt']),
    ('+ every pair rhymes (detected scheme)', lambda r: r['all_rhyme_detected']),
    ('+ no repeated end word', lambda r: r['no_repeat']),
    ('+ groups distinct (detected scheme)', lambda r: r['distinct_detected']),
]


LADDERS = (('ladder', LADDER), ('ladder_own_meter', LADDER_OWN_METER),
           ('ladder_detected', LADDER_DETECTED))


def summarise(rows):
    """Metrics for one replicate (list of analyse() dicts)."""
    n = len(rows)
    full = [r for r in rows if r['has14']]
    out = {'n': n, 'format_pct': 100 * sum(r['fmt'] for r in rows) / n}
    for k in ('orth', 'orth_quatr', 'phon', 'phon_noident', 'phon_noident_quatr',
              'phon_noident_detected'):
        g = sum(r['pairs'][k][0] for r in full)
        t = sum(r['pairs'][k][1] for r in full)
        out[f'rhyme_{k}_pct'] = 100 * g / t if t else 0.0
    nv = 14 * len(full)
    out['meter_in_range_pct'] = 100 * sum(r['meter_in_range'] for r in full) / nv if nv else 0.0
    out['meter_exact_pct'] = 100 * sum(r['meter_exact'] for r in full) / nv if nv else 0.0
    out['distinct_groups_pct'] = 100 * sum(r['distinct'] for r in full) / n
    out['distinct_groups_detected_pct'] = 100 * sum(r['distinct_detected'] for r in full) / n
    rr = [r['rr'] for r in full if 'rr' in r]
    if rr:
        t = sum(x['tot'] for x in rr)
        out['rr_coverage_pct'] = 100 * len(rr) / max(1, len(full))
        out['rr_rhyme_consonant_pct'] = 100 * sum(x['cons'] for x in rr) / t
        out['rr_rhyme_assonant_pct'] = 100 * sum(x['asso'] for x in rr) / t
        out['rr_rhyme_letters_pct'] = 100 * sum(x['letters'] for x in rr) / t
        out['rr_meter_exact_pct'] = 100 * sum(x['exact'] for x in rr) / (14 * len(rr))
        out['rr_sonnet_pct'] = 100 * sum(x['sonnet'] for x in rr) / len(rr)
    for name, ladder in LADDERS:
        alive = list(rows)
        for i, (label, pred) in enumerate(ladder):
            alive = [r for r in alive if (r['has14'] or i == 0) and pred(r)]
            out[f'{name}/{label}'] = 100 * len(alive) / n
    return out


# --------------------------------------------------------------------------
# systems
# --------------------------------------------------------------------------
def human_sample():
    lines = []
    for s in ('train', 'valid', 'test'):
        with open(os.path.join(CORPUS, f'sonnet_{s}.txt'), encoding='utf-8') as f:
            lines += [l for l in f if l.strip()]
    random.seed(HUMAN_SEED)
    # Same diaeresis sanitising as the corpus scripts: syltippy hangs on it.
    return [l.translate(str.maketrans('ïëäö', 'ieao'))
            for l in random.sample(lines, HUMAN_N)]


def well_formed(line, min_syll=7):
    """False when a verse is a fragment. About 22% of the DISCO sonnets as we
    have them split one verse in two ('mientras yo gimo con el rostro' /
    'pálido') and lose the real last verse, which shifts every later rhyme
    position; thresholds 6-8 flag the same 54-56 of the 250 human samples."""
    verses, _ = eval_sonnets_es.parse_sample(line)
    return all(rhyme_es.count_syllables_verse(v) >= min_syll for v in verses[:14])


def oracle_paper(pool, N, keep):
    """best_of_n.oracle_select: groups of N, argmax orthographic full rhyme."""
    scored = []
    for line in pool:
        s = harness.score_one(line)
        scored.append((line, s['rhyme'], s['n_lines'] == 14))
    out = []
    for i in range(0, keep * N, N):
        out.append(max(scored[i:i + N], key=lambda t: (t[1], t[2]))[0])
    return out


def oracle_strict(pool_rows, pool, N, keep):
    """Same groups, argmax identity-excluded pronunciation rhyme; ties -> distinct
    groups -> format. A selector with the evaluation's own strict criterion."""
    def score(r):
        if not r['has14']:
            return (0, False, False)
        g, t = r['pairs']['phon_noident']
        return (g / t if t else 0, r['distinct'], r['fmt'])
    out = []
    for i in range(0, keep * N, N):
        j = max(range(i, i + N), key=lambda k: score(pool_rows[k]))
        out.append(pool[j])
    return out


def load_systems(names, phon):
    """name -> list of replicates, each a list of sample lines."""
    systems = {}
    for name in names:
        base = name.partition('@')[0]   # '@<scheme>' only changes the scoring target
        if base == 'human':
            systems[name] = [human_sample()]
        elif base == 'human_clean':
            systems[name] = [[l for l in human_sample() if well_formed(l)]]
        elif base == 'haiku':
            systems[name] = [harness.load_samples(os.path.join(
                ROOT, 'results', 'llm_baseline', f'haiku45_run{i}.txt')) for i in (1, 2, 3)]
        elif base == 'haiku_single':
            # the submitted NeurIPS baseline: one subagent wrote all 30 in one conversation
            systems[name] = [harness.load_samples(os.path.join(
                ROOT, 'results', 'llm_baseline', 'haiku45_sonnets.txt'))]
        elif base == 'haiku_v2':
            # one fresh Claude Code subagent per sonnet; see haiku45_v2/PROMPT.md
            systems[name] = [harness.load_samples(os.path.join(
                ROOT, 'results', 'llm_baseline', 'haiku45_v2', f'run{i}.txt'))
                for i in (1, 2, 3)
                if os.path.exists(os.path.join(ROOT, 'results', 'llm_baseline',
                                               'haiku45_v2', f'run{i}.txt'))]
        elif base == 'dream':
            # dream_baseline_metrics.json treats the 30 samples as three ordered blocks
            d = harness.load_samples(os.path.join(
                ROOT, 'results', 'dream_baseline', 'dream7b_sonnets.txt'))
            systems[name] = [d[0:10], d[10:20], d[20:30]]
        elif base.startswith('bon'):
            m = re.fullmatch(r'bon(\d+)(_strict)?', base)
            N, strict = int(m.group(1)), bool(m.group(2))
            reps = []
            for p in sorted(glob.glob(os.path.join(SAMPLES, 'pool', 'seed*.txt.gz'))):
                with open(p[:-len('.txt.gz')] + '.json') as f:
                    meta = json.load(f)
                if N > meta['pool_N']:
                    print(f"[{name}] pool holds only {meta['pool_N']} per kept sonnet")
                    break
                pool, keep = harness.load_samples(p), meta['keep']
                if strict:
                    rows = [analyse(l, phon) for l in pool]
                    reps.append(oracle_strict(rows, pool, N, keep))
                else:
                    reps.append(oracle_paper(pool, N, keep))
            systems[name] = reps
        else:
            # base may be '<dir>' or 'ckpt100k/<dir>' (regen_samples --ckpt-step)
            files = sorted(glob.glob(os.path.join(SAMPLES, base, 'seed*.txt')))
            systems[name] = [harness.load_samples(p) for p in files]
        if not systems.get(name):
            print(f"[{name}] no samples found -- skipped")
            del systems[name]
    return systems


DEFAULT_SYSTEMS = ['human', 'human_clean', 'haiku', 'haiku_v2', 'dream', 'nocontrol', 'bon6', 'bon168',
                   'bon168_strict', 'K2', 'K5', 'K5_repel',
                   # another structure: tercets CDC DCD, scored under that target
                   'human@cdcdcd', 'human_clean@cdcdcd', 'nocontrol@cdcdcd', 'K5_cdcdcd']

HEADLINE = [('format_pct', 'Fmt'), ('rhyme_orth_pct', 'Rhyme (paper)'),
            ('rhyme_phon_pct', 'Rhyme (phon.)'),
            ('rhyme_phon_noident_pct', 'Rhyme (phon., no repeats)'),
            ('rhyme_phon_noident_detected_pct', 'Rhyme (detected scheme)'),
            ('distinct_groups_pct', 'Distinct groups'),
            ('distinct_groups_detected_pct', 'Distinct groups (detected scheme)'),
            ('meter_in_range_pct', 'Meter 10-12'), ('meter_exact_pct', 'Meter =11'),
            ('rr_rhyme_consonant_pct', 'Rantanplan consonant rhyme (no repeats)'),
            ('rr_rhyme_assonant_pct', 'Rantanplan assonant-or-better rhyme (no repeats)'),
            ('rr_rhyme_letters_pct', 'Rantanplan consonant, scheme letters (no repeats)'),
            ('rr_meter_exact_pct', 'Rantanplan meter =11'),
            ('rr_sonnet_pct', 'Rantanplan: classified as sonnet'),
            ('rr_coverage_pct', 'Rantanplan coverage')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='*', default=DEFAULT_SYSTEMS)
    ap.add_argument('--seseo', action='store_true')
    ap.add_argument('--yeismo', action='store_true')
    ap.add_argument('--samples', default=SAMPLES,
                    help='root of <config>/seed<k>.txt (regen_samples --out)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'strict_metrics.json'))
    ap.add_argument('--dump-lines', default=None,
                    help='write every sonnet of every system to this file for '
                         'rantanplan_score.py, then stop')
    a = ap.parse_args()
    globals()['SAMPLES'] = a.samples

    phon = make_phon_key(a.seseo, a.yeismo)
    systems = load_systems(a.systems, phon)
    if a.dump_lines:
        seen = set()
        with open(a.dump_lines, 'w', encoding='utf-8') as f:
            for reps in systems.values():
                for l in (l for rep in reps for l in rep):
                    if rr_key(l) not in seen:
                        seen.add(rr_key(l))
                        f.write(l.strip() + '\n')
        print(f"{len(seen)} unique sonnets -> {a.dump_lines}")
        return
    load_rantanplan()
    results = {}
    for name, reps in systems.items():
        target = split_target(name)[1]
        rows = [[analyse(l, phon, target) for l in rep] for rep in reps]
        per_rep = [summarise(rr) for rr in rows]
        keys = [k for k in per_rep[0] if k != 'n' and all(k in r for r in per_rep)]
        schemes = {}
        for r in (r for rr in rows for r in rr if r['has14']):
            schemes[r['scheme']] = schemes.get(r['scheme'], 0) + 1
        results[name] = dict(
            replicates=len(reps), n_per_replicate=[r['n'] for r in per_rep],
            per_replicate=[{k: round(v, 2) for k, v in r.items()} for r in per_rep],
            aggregate=harness.aggregate_seeds(per_rep, keys),
            detected_schemes=dict(sorted(schemes.items(), key=lambda kv: -kv[1])))
        print(f"[{name}] {len(reps)} replicate(s), n={[r['n'] for r in per_rep]}",
              flush=True)

    out = dict(phon_key=dict(seseo=a.seseo, yeismo=a.yeismo), systems=results)
    with open(a.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    def cell(name, k):
        m = results[name]['aggregate'].get(k)
        if m is None:
            return '—'
        return (f"{m['mean']:.1f}" if m['n'] == 1 or m['halfwidth'] == 0
                else f"{m['mean']:.1f} ± {m['halfwidth']:.1f}")

    names = list(results)
    print('\n| Metric | ' + ' | '.join(names) + ' |')
    print('|---' * (len(names) + 1) + '|')
    for k, label in HEADLINE:
        print(f'| {label} | ' + ' | '.join(cell(n, k) for n in names) + ' |')
    for title, ladder in LADDERS:
        print(f'\n| Cumulative pass, % of poems ({title}) | ' + ' | '.join(names) + ' |')
        print('|---' * (len(names) + 1) + '|')
        for label, _ in ladder:
            print(f'| {label} | ' + ' | '.join(cell(n, f'{title}/{label}') for n in names) + ' |')
    print('\nGuardado:', a.out)


if __name__ == '__main__':
    main()
