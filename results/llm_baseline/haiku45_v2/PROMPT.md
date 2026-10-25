# Haiku 4.5 baseline, v2 (regenerated 2026-10-01)

**Why v2:** the v1 baseline in the submitted paper (`../haiku45_sonnets.txt`) was produced by a single
Claude Code subagent that wrote all 30 sonnets in one conversation, and quality decayed over it
(rhyme by block of 10: 60.7 / 61.3 / 22.0). v2 gives every sonnet its own fresh context.

**Model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), accessed through Claude Code subagents
(`model: haiku`, `subagent_type: general-purpose`), not the raw API: no API credits were available.
Each subagent therefore also sees Claude Code's own system prompt **and the user's global CLAUDE.md**,
which includes the instruction to write neutral (non-Argentine) Spanish. At least one subagent's report
cites it ("Español neutral (sin argentinismos)"). Sampling settings are Claude Code's defaults and are not
controllable. Both points must be disclosed wherever this baseline is reported.

**Protocol:** 3 runs × 30 topics = 90 subagents, one sonnet each, one fresh context each. Topics are
cycled in the fixed order of `docs/baseline_prompts.md`.

**Prompt** (the task paragraph is the submitted paper's template verbatim; the second paragraph is a
protocol wrapper added because inside Claude Code the pilot subagent answered with a verification
report and used a tool):

```
Escribe un soneto en español. Debe tener exactamente 14 versos, cada verso de once sílabas (endecasílabo), con esquema de rima ABBA ABBA CDE CDE. Tema: {tema}. Responde únicamente con los 14 versos, uno por línea, sin título ni comentarios.

No uses herramientas. Escribe los versos entre las etiquetas <soneto> y </soneto>.
```

**Files:** `raw/run{r}_{i:02d}_{tema}.txt` is each subagent's full reply; `run{r}.txt` is the
`<eos>` evaluation format, built with `experiments/dream_baseline.py::to_eval_line` (the shared
normalization: lowercase, punctuation stripped, accents kept) from the text between the tags.
Raw line counts (before truncation to 14) go to `run{r}_raw_verse_counts.json`.

**Pilot (discarded, not in any run):** topic *mar*, template prompt without the wrapper. The reply
began with a verification checklist before the verses.

**Provenance:** `provenance.tsv` records, for each of the 90 subagents, the model that wrote the
extracted answer (all `claude-haiku-4-5-20251001`) and a sha256 of its transcript, written by
`experiments/collect_haiku_v2.py --provenance`. The transcripts stay in the Claude Code session
directory and are not copied into the repo, because they contain local paths and the user's global
CLAUDE.md.
