"""Blind LLM-judge fluency ratings with local models (EvoMUSART resubmission).

Two judges from different families, neither a compared system (Claude Haiku
is a baseline, so no Claude judge); see paper/evomusart/EVOMUSART_2027.md:
  qwen38   unsloth/Qwen3.8-27B-GGUF      Qwen3.8-27B-UD-Q4_K_M.gguf
  gemma4   unsloth/gemma-4-31B-it-GGUF   gemma-4-31B-it-Q4_K_M.gguf

Each judge runs behind llama.cpp's llama-server (OpenAI-compatible API), using
the CUDA build that Unsloth Studio installs (b11139, sm_86 supported):
  ~/.unsloth/llama.cpp/build/bin/llama-server -m <gguf> -ngl 99 -c 4096 --jinja --port 8080
The GGUF files live in the Hugging Face cache (hf_hub_download). Do not run
Unsloth Studio on the GPU at the same time.

Protocol:
  * blind: the judge sees only the poem; poems from all systems are shuffled
    together with a fixed seed;
  * same normalisation as fluency_ppl.py (lowercase, letters only, one verse
    per line), and the prompt says so, so punctuation cannot decide;
  * the prompt says poems may imitate Golden Age Spanish, so hyperbaton and
    archaisms are not read as errors (without it Qwen3.8 rated 23% of the
    human sonnets <= 2 for grammar or coherence, citing verb inversions;
    those first-round results are in results/fluency/v1_no_period_note/);
  * rubric 1-5 for grammaticality, coherence and poetic quality; rhyme and
    meter are excluded (measured elsewhere, and LLMs count syllables badly);
  * a short justification BEFORE the scores, JSON enforced by a schema;
  * greedy decoding (temperature 0), thinking disabled.

Pairwise mode (--pairs): on our model's outputs the absolute scale has a
floor -- in a 30-poem check every no-control, K5 and best-of-168 sample got 1
for grammaticality and coherence -- so it cannot say whether control helps or
hurts. Pairwise preference still discriminates among weak texts: poem i of
system X is compared with poem i of system Y within the same seed, asked in
both orders (position bias), and each criterion is answered A, B or igual.
A system's preference rate is wins + half the ties over both orders.

Uso:
  conda run -n gpu python experiments/llm_judge.py --judge qwen38      # server must be up
  conda run -n gpu python experiments/llm_judge.py --judge qwen38 --pairs
  conda run -n gpu python experiments/llm_judge.py --report
Salida: results/fluency/judge_<judge>.jsonl, results/fluency/judge.json (+ table)
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import fluency_ppl as fp
import harness

OUT = os.path.join(ROOT, 'results', 'fluency')
JUDGES = {
    'qwen38': dict(gguf='Qwen3.8-27B-UD-Q4_K_M.gguf',
                   extra={'chat_template_kwargs': {'enable_thinking': False}}),
    # Studio's template thinks by default; without this the JSON never arrives
    'gemma4': dict(gguf='gemma-4-31B-it-Q4_K_M.gguf',
                   extra={'chat_template_kwargs': {'enable_thinking': False}}),
}
CRITERIA = ['gramaticalidad', 'coherencia', 'calidad_poetica']
SHUFFLE_SEED = 20261017

SYSTEM_PROMPT = "Eres un evaluador experto en poesía en español."
USER_PROMPT = """A continuación hay un poema de catorce versos, uno por línea. En todos los poemas evaluados se eliminaron las mayúsculas y los signos de puntuación: no los consideres. Los poemas pueden imitar la lengua del Siglo de Oro (hipérbaton, arcaísmos, ortografía antigua): no penalices esos rasgos por sí mismos. Tampoco evalúes la rima ni la métrica, que se miden por separado.

Evalúa el poema de 1 a 5 en cada criterio:
- gramaticalidad: si los versos forman construcciones gramaticalmente correctas en español.
- coherencia: si el poema tiene sentido y mantiene un tema o hilo reconocible.
- calidad_poetica: si el lenguaje tiene valor poético (imágenes, recursos expresivos, tono).

Escala: 1 = ininteligible o muy deficiente; 3 = aceptable, con problemas claros; 5 = excelente, propio de un poeta competente.

Primero escribe una justificación breve (una o dos oraciones) y después los puntajes.

<poema>
{poem}
</poema>"""

SCHEMA = {
    'type': 'object',
    'properties': {
        'justificacion': {'type': 'string'},
        **{c: {'type': 'integer', 'minimum': 1, 'maximum': 5} for c in CRITERIA},
    },
    'required': ['justificacion'] + CRITERIA,
}


PAIRS = [('K5', 'nocontrol'), ('K5', 'bon168'), ('bon168', 'nocontrol'),
         ('K5_repel', 'K5'), ('K5_repel', 'nocontrol'),
         ('K5_cdcdcd', 'nocontrol')]

PAIR_PROMPT = """A continuación hay dos poemas de catorce versos, A y B, uno por línea. En ambos se eliminaron las mayúsculas y los signos de puntuación: no los consideres. Los poemas pueden imitar la lengua del Siglo de Oro (hipérbaton, arcaísmos, ortografía antigua): no penalices esos rasgos por sí mismos. Tampoco compares la rima ni la métrica, que se miden por separado.

Para cada criterio, indica cuál de los dos poemas es mejor (A o B), o "igual" si no hay una diferencia apreciable:
- gramaticalidad: cuál forma construcciones más correctas en español.
- coherencia: cuál tiene más sentido y mantiene mejor un tema o hilo.
- calidad_poetica: cuál tiene un lenguaje con más valor poético.

Los dos poemas pueden ser deficientes; compáralos entre sí. Primero escribe una justificación breve (una o dos oraciones) y después las elecciones.

<poema_A>
{a}
</poema_A>

<poema_B>
{b}
</poema_B>"""

PAIR_SCHEMA = {
    'type': 'object',
    'properties': {
        'justificacion': {'type': 'string'},
        **{c: {'type': 'string', 'enum': ['A', 'B', 'igual']} for c in CRITERIA},
    },
    'required': ['justificacion'] + CRITERIA,
}


def ask(url, poem, extra, prompt=None, schema=None):
    body = dict(
        messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                  {'role': 'user', 'content': prompt or USER_PROMPT.format(poem=poem)}],
        temperature=0.0, max_tokens=400, seed=0,
        response_format={'type': 'json_schema',
                         'json_schema': {'name': 'rating', 'schema': schema or SCHEMA}},
        **extra)
    req = urllib.request.Request(url + '/v1/chat/completions',
                                 data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return json.loads(out['choices'][0]['message']['content']), out.get('model')


def judge(name, url):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'judge_{name}.jsonl')
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            done = {json.loads(l)['key'] for l in f if l.strip()}
    texts = sorted({t for reps in fp.systems_texts().values() for rep in reps for t in rep if t})
    random.Random(SHUFFLE_SEED).shuffle(texts)
    todo = [t for t in texts if fp.key(t) not in done]
    print(f"[{name}] {len(todo)} poems to judge, {len(done)} cached", flush=True)
    t0 = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for i, t in enumerate(todo, 1):
            rating, served = ask(url, t, JUDGES[name]['extra'])
            f.write(json.dumps(dict(key=fp.key(t), served_model=served, **rating),
                               ensure_ascii=False) + '\n')
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  {(time.time()-t0)/i:.1f}s/poem", flush=True)


def pair_texts():
    """{(X, Y): [(seed_index, i, text_X, text_Y)]} for the PAIRS present."""
    names = sorted({n for p in PAIRS for n in p})
    old = fp.SYSTEMS
    fp.SYSTEMS = names
    texts = fp.systems_texts()
    fp.SYSTEMS = old
    out = {}
    for x, y in PAIRS:
        if x not in texts or y not in texts:
            continue
        out[(x, y)] = [(r, i, tx, ty)
                       for r, (rx, ry) in enumerate(zip(texts[x], texts[y]))
                       for i, (tx, ty) in enumerate(zip(rx, ry)) if tx and ty]
    return out


def judge_pairs(name, url):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'pairs_{name}.jsonl')
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            done = {json.loads(l)['key'] for l in f if l.strip()}
    jobs = []
    for (x, y), items in pair_texts().items():
        for r, i, tx, ty in items:
            for order in ('XY', 'YX'):
                k = f'{x}|{y}|{fp.key(tx)}|{fp.key(ty)}|{order}'
                if k not in done:
                    jobs.append((k, x, y, r, i, tx, ty, order))
    random.Random(SHUFFLE_SEED).shuffle(jobs)
    print(f"[{name} pairs] {len(jobs)} comparisons to run, {len(done)} cached", flush=True)
    t0 = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for n, (k, x, y, r, i, tx, ty, order) in enumerate(jobs, 1):
            a, b = (tx, ty) if order == 'XY' else (ty, tx)
            res, served = ask(url, None, JUDGES[name]['extra'],
                              prompt=PAIR_PROMPT.format(a=a, b=b), schema=PAIR_SCHEMA)
            # Translate A/B back to X/Y.
            to_xy = {'A': order[0], 'B': order[1], 'igual': '='}
            f.write(json.dumps(dict(key=k, x=x, y=y, rep=r, i=i, order=order,
                                    served_model=served,
                                    justificacion=res['justificacion'],
                                    **{c: to_xy[res[c]] for c in CRITERIA}),
                               ensure_ascii=False) + '\n')
            f.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}  {(time.time()-t0)/n:.1f}s/comparison", flush=True)


def report_pairs():
    # repetition confound: the controller raises repetition, and a judge may
    # read a narrow vocabulary as thematic coherence
    rep_of = {fp.key(t): fp.diversity(t)['repetition']
              for items in pair_texts().values() for _, _, tx, ty in items for t in (tx, ty)}
    out = {}
    for j in JUDGES:
        p = os.path.join(OUT, f'pairs_{j}.jsonl')
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for x, y in PAIRS:
            rs = [r for r in rows if r['x'] == x and r['y'] == y]
            if not rs:
                continue
            res = {}
            for c in CRITERIA:
                per_rep = {}
                for r in rs:
                    v = 1.0 if r[c] == 'X' else 0.0 if r[c] == 'Y' else 0.5
                    per_rep.setdefault(r['rep'], []).append(v)
                reps = [dict(rate=sum(v) / len(v)) for v in per_rep.values()]
                agg = harness.aggregate_seeds(reps, ['rate'])['rate']
                # position consistency: same winner in both orders
                pairs = {}
                for r in rs:
                    pairs.setdefault((r['rep'], r['i']), {})[r['order']] = r[c]
                both = [v for v in pairs.values() if len(v) == 2]
                consistent = sum(v['XY'] == v['YX'] for v in both) / max(1, len(both))
                res[c] = dict(rate=agg, position_consistency=consistent,
                              n_comparisons=len(rs))
            split = {True: [], False: []}
            for r in rs:
                kx, ky = r['key'].split('|')[2:4]
                v = [1.0 if r[c] == 'X' else 0.0 if r[c] == 'Y' else 0.5 for c in CRITERIA]
                split[rep_of[kx] > rep_of[ky]].append(sum(v) / len(v))
            res['by_repetition'] = {
                ('x_more_repetitive' if k else 'x_less_repetitive'):
                    dict(rate=sum(v) / len(v) if v else None, n_comparisons=len(v))
                for k, v in split.items()}
            out[f'{j}: {x} vs {y}'] = res
    with open(os.path.join(OUT, 'pairs.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('\n| Judge: X vs Y | ' + ' | '.join(f'P(X better) {c}' for c in CRITERIA)
          + ' | position consistency | P(X better), X more / less repetitive (n) |')
    print('|---' * (len(CRITERIA) + 3) + '|')
    for k, res in out.items():
        cells = [f"{res[c]['rate']['mean']:.2f} ± {res[c]['rate']['halfwidth']:.2f}" for c in CRITERIA]
        pc = sum(res[c]['position_consistency'] for c in CRITERIA) / len(CRITERIA)
        br = res['by_repetition']
        fmt = lambda d: '—' if d['rate'] is None else f"{d['rate']:.2f} ({d['n_comparisons']})"
        print(f'| {k} | ' + ' | '.join(cells) + f' | {pc:.2f} | '
              f"{fmt(br['x_more_repetitive'])} / {fmt(br['x_less_repetitive'])} |")


def report():
    ratings = {}
    for j in JUDGES:
        p = os.path.join(OUT, f'judge_{j}.jsonl')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                ratings[j] = {r['key']: r for r in (json.loads(l) for l in f if l.strip())}
    results = {}
    for name, reps in fp.systems_texts().items():
        per_rep = []
        for rep in reps:
            rep = [t for t in rep if t]
            out = {}
            for j, rs in ratings.items():
                rows = [rs[fp.key(t)] for t in rep if fp.key(t) in rs]
                if len(rows) != len(rep):
                    continue
                for c in CRITERIA:
                    out[f'{j}_{c}'] = sum(r[c] for r in rows) / len(rows)
            per_rep.append(out)
        keys = [k for k in per_rep[0] if all(k in r for r in per_rep)]
        if keys:
            results[name] = dict(aggregate=harness.aggregate_seeds(per_rep, keys),
                                 per_replicate=per_rep)
    # Agreement between judges, per poem and criterion (Spearman).
    agreement = {}
    if len(ratings) == 2:
        from scipy.stats import spearmanr
        a, b = (ratings[j] for j in JUDGES)
        common = sorted(set(a) & set(b))
        for c in CRITERIA:
            if len(common) > 2:
                rho, _ = spearmanr([a[k][c] for k in common], [b[k][c] for k in common])
                agreement[c] = dict(spearman=rho, n=len(common))
    with open(os.path.join(OUT, 'judge.json'), 'w') as f:
        json.dump(dict(systems=results, agreement=agreement), f, indent=1, ensure_ascii=False)
    cols = sorted({k for r in results.values() for k in r['aggregate']})
    print('| System | ' + ' | '.join(cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for name, r in results.items():
        cells = []
        for k in cols:
            m = r['aggregate'].get(k)
            cells.append('—' if m is None else f"{m['mean']:.2f} ± {m['halfwidth']:.2f}")
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    if agreement:
        print('\nJudge agreement (Spearman over poems):',
              {c: round(v['spearman'], 3) for c, v in agreement.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', choices=list(JUDGES))
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--pairs', action='store_true', help='pairwise mode (with --judge)')
    ap.add_argument('--systems', nargs='*', default=None)
    a = ap.parse_args()
    if a.systems:
        fp.SYSTEMS = a.systems
    if a.judge and a.pairs:
        judge_pairs(a.judge, a.url)
    elif a.judge:
        judge(a.judge, a.url)
    if a.report:
        report()
        report_pairs()
