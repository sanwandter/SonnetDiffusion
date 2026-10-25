"""Consolidate all experiment JSONs into the tables the paper needs.

Plan B (the LaTeX rewrite) reads results/tablas_paper.md and nothing else,
so there is exactly one place where a number can be wrong.

Two deviations from the plan's sketch, both forced by the data as it actually
came out:

  * The plan read `bon['uncontrolled']` and `bon['N']`. Neither exists -- Task 4
    became a sweep over several N rather than a single matched point. The
    uncontrolled row is `best_of_n['1']`: oracle selection over groups of one
    is no selection at all, so best-of-1 IS the plain sampling arm, measured
    rather than run twice.
  * The main table carries syllable and end-word columns. The controlled model
    loses both to the prompted baselines, and a rhyme-only table would be
    hiding it. The cost column is in the same table for the same reason: the
    method's claim is per unit compute, so the price belongs beside the win.

Missing inputs are reported and skipped, so this is safe to run before every
experiment has finished.

Uso: conda run -n gpu python experiments/build_tables.py
Salida: results/paper_tables.json + results/tablas_paper.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATOS = os.path.join(ROOT, 'results')

MISSING = []


def load(name):
    path = os.path.join(DATOS, name)
    if not os.path.exists(path):
        print(f"  MISSING: {name}")
        MISSING.append(name)
        return None
    with open(path) as f:
        return json.load(f)


def fmt(agg, key):
    """mean +/- half-width, or a dash when the input is absent."""
    if agg is None or key not in agg:
        return "--"
    a = agg[key]
    return f"{a['mean']:.1f} $\\pm$ {a['halfwidth']:.1f}"


def main():
    audit = load('audit_measurements.json')
    bon = load('best_of_n_results.json')
    abl = load('ablation_seeded.json')
    ckpt = load('checkpoint_progression.json')
    llm = load('llm_baseline_metrics.json')
    dream = load('dream_baseline_metrics.json')
    ksweep = load('k_sweep_results.json')

    L = ["# Tablas del paper (generado por build_tables.py)", "",
         "Numeros: media +/- semiancho del IC al 95% (t de Student) sobre 3 "
         "semillas. No editar a mano: regenerar con build_tables.py.", ""]

    # ---------------------------------------------------------------- T1
    L += ["## T1 - Comparacion principal", "",
          "| Sistema | Params | s/soneto | Formato % | Rima % | "
          "Silabas en [10,12] % | Finales distintos % |",
          "|---|---|---|---|---|---|---|"]

    def row(label, params, cost, agg, fmt_override=None):
        f_ = fmt_override if fmt_override else fmt(agg, 'format_pct')
        L.append(f"| {label} | {params} | {cost} | {f_} | "
                 f"{fmt(agg, 'rhyme_full_pct')} | "
                 f"{fmt(agg, 'syll_in_range_pct')} | "
                 f"{fmt(agg, 'distinct_endwords_pct')} |")

    # Human corpus. Rhyme is 100% on both the full scheme and the quartets,
    # validated separately by the author, and is a constant here rather than
    # something recomputed: RHYME_GROUPS accepts only the CDE CDE tercet
    # arrangement and scores every other standard Spanish scheme as wrong,
    # which is where the retired 74.1% came from. Syllables and end words do
    # not reference the rhyme scheme, so those are measured.
    corpus = load('corpus_meter_stats.json')
    if corpus:
        L.append(f"| Corpus humano (muestra n={corpus['n']}) | -- | -- | "
                 f"{corpus['format_pct']:.1f} | 100.0 | "
                 f"{corpus['syll_in_range_pct']:.1f} | "
                 f"{corpus['distinct_endwords_pct']:.1f} |")
    else:
        L.append("| Corpus humano | -- | -- | 100.0 | 100.0 | -- | -- |")

    if llm:
        # format_pct is 100 by construction -- the prompt demanded 14 verses,
        # so the metric measures the prompt, not the model. Never print it.
        row('Claude Haiku 4.5 (prompted)', 'n/d', 'n/d', llm['aggregate'],
            fmt_override='n/a (instruido)')
    if dream:
        counts = load('dream_baseline/dream7b_raw_verse_counts.json')
        measured = (f"{counts['format_pct_measured']:.1f}" if counts
                    else 'n/a (instruido)')
        row('Dream-7B-Instruct (prompted)', '7B', 'n/d', dream['aggregate'],
            fmt_override=measured)
    if bon:
        p = f"{audit['params_millions']}M" if audit else "?"
        arms = [('1', 'Nuestro, sin control')]
        for N in (bon['n_forward_matched'], bon['n_wallclock_matched']):
            key = str(N)
            if key in bon['best_of_n']:
                tag = ('igualado en forward passes'
                       if N == bon['n_forward_matched'] else 'igualado en wall-clock')
                arms.append((key, f"Nuestro, best-of-{N} oraculo ({tag})"))
        for key, label in arms:
            e = bon['best_of_n'][key]
            row(label, p, f"{e['seconds_per_sonnet']:.2f}", e['aggregate'])
        c = bon['controlled']
        row('**Nuestro, control por gradiente**', p,
            f"{c['seconds_per_sonnet']:.2f}", c['aggregate'])
    L.append("")

    # ---------------------------------------------------------------- T2
    if bon:
        L += ["## T2 - Barrido completo de best-of-N (oraculo)", "",
              "Seleccion por oraculo: cada grupo de N se puntua con la misma "
              "metrica de rima que reporta el paper y se queda el mejor. La "
              "linea base hace trampa a proposito.", "",
              "| N | s/soneto | Rima % |", "|---|---|---|"]
        for N in bon['N_values']:
            e = bon['best_of_n'][str(N)]
            L.append(f"| {N} | {e['seconds_per_sonnet']:.2f} | "
                     f"{fmt(e['aggregate'], 'rhyme_full_pct')} |")
        c = bon['controlled']
        L.append(f"| control por gradiente | {c['seconds_per_sonnet']:.2f} | "
                 f"{fmt(c['aggregate'], 'rhyme_full_pct')} |")
        L.append("")

    # ---------------------------------------------------------------- T3
    if abl:
        L += ["## T3 - Ablacion de controladores", "",
              "| Config | Formato % | Rima % | Silabas en [10,12] % | "
              "Finales distintos % |", "|---|---|---|---|---|"]
        for r in abl:
            L.append(f"| {r['config']} | {fmt(r['aggregate'], 'format_pct')} | "
                     f"{fmt(r['aggregate'], 'rhyme_full_pct')} | "
                     f"{fmt(r['aggregate'], 'syll_in_range_pct')} | "
                     f"{fmt(r['aggregate'], 'distinct_endwords_pct')} |")
        L.append("")

    # ---------------------------------------------------------------- T4
    if ksweep:
        L += ["## T4 - Barrido de pasos internos K", "",
              "| K | Formato % | Rima % | s/soneto | forward passes |",
              "|---|---|---|---|---|"]
        for r in ksweep:
            L.append(f"| {r['K']} | {fmt(r['aggregate'], 'format_pct')} | "
                     f"{fmt(r['aggregate'], 'rhyme_full_pct')} | "
                     f"{r['seconds_per_sonnet']:.2f} | "
                     f"{r['forward_calls']['mean']:.0f} |")
        L.append("")

    # ------------------------------------------------------- lambda sweeps
    # Whole curves, never the argmax. These weights were inherited rather than
    # tuned, which is what makes them honest; selecting the best value on the
    # metric the paper reports would be tuning on the test set.
    for param, title, note in [
        ('lambda_rep', 'T4b - Barrido de lambda_rep',
         'La ablacion solo comparo 0.2 contra 0.0, donde el efecto es '
         'indistinguible del ruido. El barrido muestra que la penalizacion no '
         'esta rota sino sub-ponderada: en 1.0 recupera 7 puntos de finales '
         'distintos sin costo significativo en rima.'),
        ('lambda_rhyme', 'T4c - Barrido de lambda_rhyme',
         'Frontera entre rima y diversidad lexica. La rima satura cerca de '
         '1.0; los finales distintos caen de forma monotona.'),
    ]:
        sw = load(f'sweep_{param}.json')
        if not sw:
            continue
        L += [f"## {title}", "", note, "",
              f"| {param} | Rima % | Silabas en [10,12] % | Finales distintos % |",
              "|---|---|---|---|"]
        for r in sw['points']:
            mark = ' *' if r['value'] == sw['default'] else ''
            L.append(f"| {r['value']}{mark} | "
                     f"{fmt(r['aggregate'], 'rhyme_full_pct')} | "
                     f"{fmt(r['aggregate'], 'syll_in_range_pct')} | "
                     f"{fmt(r['aggregate'], 'distinct_endwords_pct')} |")
        L += ["", "\\* valor heredado, usado en todos los demas experimentos.", ""]

    # ---------------------------------------------------------------- T5
    if ckpt:
        L += ["## T5 - Progresion por checkpoint (apendice)", "",
              "Solo los checkpoints EMA que siguen en disco. Los pasos "
              "55k-70k del paper original no son reproducibles: esos archivos "
              "no estan en el repositorio.", "",
              "| Paso | Formato % | Rima % |", "|---|---|---|"]
        for r in ckpt:
            L.append(f"| {r['step']} | {fmt(r['aggregate'], 'format_pct')} | "
                     f"{fmt(r['aggregate'], 'rhyme_full_pct')} |")
        L.append("")

    # ---------------------------------------------------------------- costs
    if audit:
        L += ["## Costos medidos", ""]
        # Wall-clock cost per sonnet depends on batch size: the controlled path
        # holds an autograd graph over K inner steps and materialises a
        # [B, 256, 23001] logit tensor, so a batch of 30 costs more than twice
        # as much PER SONNET as a batch of 8. The figures that matter are the
        # ones measured at the batch size the experiments used; the audit's are
        # kept only to document where the earlier mismatched budget came from.
        b = bon.get('budget_source', {}) if bon else {}
        if b.get('measured_seconds_per_sonnet_controlled'):
            L += [f"Medidos con lote de {bon['n_per_seed']}, el mismo de todos "
                  f"los experimentos:", "",
                  f"- Sin control: "
                  f"{b['measured_seconds_per_sonnet_uncontrolled']} s/soneto, "
                  f"{audit['uncontrolled']['forward_calls']['mean']:.0f} forward passes",
                  f"- Con control: "
                  f"{b['measured_seconds_per_sonnet_controlled']} s/soneto, "
                  f"{audit['controlled']['forward_calls']['mean']:.0f} forward passes",
                  f"- Razon wall-clock: **{b['measured_ratio_walltime']}x**",
                  f"- Razon forward passes: **{audit['ratio_forward_calls']}x** "
                  f"(independiente del lote)", ""]
        L += [f"- Parametros: {audit['params_millions']}M "
              f"({audit['params_total']:,} exactos)",
              "- Tiempos medidos solo sobre el bucle de muestreo, sin contar "
              "la carga del modelo.",
              f"- Auditoria previa con lote de {audit['n']} (NO comparable con "
              f"lo anterior): {audit['seconds_per_sonnet_uncontrolled']} y "
              f"{audit['seconds_per_sonnet_controlled']} s/soneto, "
              f"razon {audit['ratio_walltime']:.1f}x. De aqui salio el "
              f"best-of-69 mal igualado.", ""]

    if MISSING:
        L += ["## Faltantes", "",
              "Tablas omitidas por falta de datos:", ""]
        L += [f"- `{m}`" for m in MISSING]
        L.append("")

    md = '\n'.join(L)
    with open(os.path.join(ROOT, 'results', 'tablas_paper.md'), 'w') as f:
        f.write(md)
    with open(os.path.join(DATOS, 'paper_tables.json'), 'w') as f:
        json.dump(dict(audit=audit, best_of_n=bon, ablation=abl,
                       checkpoints=ckpt, llm=llm, dream=dream,
                       k_sweep=ksweep), f, indent=1)
    print(md)


if __name__ == '__main__':
    main()
