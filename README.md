# SonnetDiffusion: code, model, samples and measurements

Anonymous repository for the paper "SonnetDiffusion: Training-Free Analytic Control for Formal
Constraints in Diffusion Poetry Generation". It holds the code, the model checkpoint, every
generated sample the paper scores, the raw measurements behind each table, and the supplementary
material.

## Start here

- `supplement/SUPPLEMENT.md`: supplementary tables and figure (Tables S1-S8, Figure S1), the full
  penalty definitions, the baseline prompt, and licenses.
- `supplement/best_sonnets.md`: a subset of five sonnets per configuration, with their scores next to
  the configuration means. All 90 samples per configuration are in `results/samples/`.

## Quick start

Run everything from the repository root. Generation needs a CUDA GPU (24 GB is enough).

```bash
conda create -n sonnetdiffusion python=3.12 && conda activate sonnetdiffusion
pip install -r requirements.txt

# Generate sonnets with the paper's controller (K = 5), n per seed, seeds 102 / 202 / 302.
# Each sonnet takes about 70 s; without control (nocontrol) under 1 s.
python experiments/regen_samples.py --only K5 --n 4 --pool-N 2 --out my_samples

# Score them: format, rhyme and meter (prints a few samples and the metrics)
python sonnetdiffusion/eval_sonnets_es.py my_samples/K5/seed102.txt
```

`--only` takes any configuration of the paper: `nocontrol`, `K2`, `K5`, `K5_repel`, `K5_cdcdcd`.
Each output line is one sonnet, verses separated by `< eos >`. The controller and its weights are
defined in `experiments/harness.py` (`FULL`) and `sonnetdiffusion/controllers_es.py`; the target
rhyme scheme is a dictionary of line groups (see `K5_cdcdcd` in `experiments/regen_samples.py`).

## Layout

| Path | Contents |
|---|---|
| `sonnetdiffusion/` | Controller penalties (`controllers_es.py`), rhyme suffixes and syllable counter (`rhyme_es.py`), evaluator (`eval_sonnets_es.py`), vocabulary lists (`vocab_es.py`) |
| `improved-diffusion/` | Diffusion-LM backbone (Li et al., 2022) with the sampling hook used by the controller; training script |
| `models/diffusion/diff_e2e-tgt_pad_2000steps_16dim_sonnet_es/` | EMA checkpoint at step 50k (`ema_0.9999_050000.pt`, with SHA-256), vocabulary, embedding initialization, training arguments |
| `models/diffusion/diff_e2e-tgt_pad_2000steps_16dim_sonnet_es.sh` | Training command |
| `data/` | Corpus preparation script and the train / validation / test splits |
| `experiments/` | One driver per experiment; `harness.py` holds seeding, generation, scoring and intervals |
| `results/` | Aggregate measurements (JSON), raw samples, caches of every external score |
| `supplement/` | Supplementary material |

## Paper tables and where they come from

| Paper | Script(s) | Result file(s) |
|---|---|---|
| Table 1, main results | `k_sweep.py`, `best_of_n.py`, `score_llm_baseline.py`, `dream_baseline.py`, `corpus_meter_stats.py`, `build_tables.py` | `results/k_sweep_results.json`, `results/best_of_n_results.json`, `results/llm_baseline_metrics.json`, `results/dream_baseline_metrics.json`, `results/corpus_meter_stats.json`, `results/paper_tables.json` |
| Table 2, ablation | `ablation_seeded.py` | `results/ablation_seeded.json` |
| Table 3, sample pair | (samples) | `results/piezas/soneto_50k_controlado.txt`, `results/piezas/soneto_50k_libre.txt` |
| Table 4, strict and independent rhyme | `regen_samples.py`, `strict_metrics.py`, `rantanplan_score.py` | `results/samples/`, `results/strict_metrics.json`, `results/strict_metrics.md`, `results/rantanplan_cache_v2.jsonl` |
| Table 5, fluency | `fluency_ppl.py`, `llm_judge.py` | `results/fluency/` |
| Cost measurements (Section 5) | `audit_measurements.py` | `results/audit_measurements.json` |
| Tables S1-S2, Figure S1 | `k_sweep.py`, `best_of_n.py` | `results/k_sweep_results.json`, `results/best_of_n_results.json` |
| Table S3, weight sweeps | `lambda_sweep.py` | `results/sweep_lambda_rep.json`, `results/sweep_lambda_rhyme.json` |
| Table S4, concept and emotion words | (counted on the samples) | `results/samples/` |
| Tables S5-S6, fluency details | `llm_judge.py --report`, `fluency_ppl.py --report` | `results/fluency/pairs_gemma4.jsonl`, `results/fluency/fluency.md` |

Every script documents its usage in its docstring. Run them from the repository root, for example
`python experiments/k_sweep.py --n 30`.

## Samples

`results/samples/<configuration>/seed<s>.txt` holds 30 sonnets per seed (seeds 102, 202, 302), one
per line, verses separated by `< eos >`, exactly as the model emits them (lowercase, no
punctuation). Configurations: `nocontrol`, `K2`, `K5`, `K5_repel` (cross-group repulsion), `K5_cdcdcd`
(CDC DCD tercet target), and `pool` (the uncontrolled candidate pool behind oracle best-of-N,
gzipped). Each `seed<s>.json` records the configuration, timing and the metrics at generation time.

The prompted baselines are in `results/llm_baseline/haiku45_sonnets.txt` (Claude Haiku 4.5) and
`results/dream_baseline/` (Dream-7B-Instruct).

## External scores

Perplexity (Salamandra-7B, Gemma-4-12B) and the Gemma-4-31B-it judge are cached per poem in
`results/fluency/cache_*.jsonl`, `judge_gemma4.jsonl` and `pairs_gemma4.jsonl`, keyed by the SHA-1 of
the normalized poem text, so the reports can be rebuilt without the models. Rantanplan output is
cached the same way in `results/rantanplan_cache_v2.jsonl`. The paper reports the Gemma judge only.

## Environment

Tested with Python 3.12, PyTorch 2.14 (CUDA 13.0), Transformers 5.16, NumPy 2.2 and SciPy 1.18, on
one 24 GB GPU. `requirements.txt` lists the Python packages.

- **Rantanplan** needs spaCy 2.2 and Python 3.8, so it runs in its own environment; setup and usage
  are in the docstring of `experiments/rantanplan_score.py`.
- **The LLM judge** runs behind llama.cpp's `llama-server`; setup is in the docstring of
  `experiments/llm_judge.py`.
- **Training** (`models/diffusion/diff_e2e-tgt_pad_2000steps_16dim_sonnet_es.sh`) runs from
  `improved-diffusion/` with the backbone on the path:
  `cd improved-diffusion && PYTHONPATH=. sh ../models/diffusion/diff_e2e-tgt_pad_2000steps_16dim_sonnet_es.sh`.
  It also needs `wandb` and `blobfile`.

## Licenses

DISCO corpus: CC-BY 4.0. Golden Age Sonnets corpus annotations: CC-BY-NC 4.0. Dream-7B-Instruct,
Salamandra-7B, Gemma-4 and Rantanplan: Apache 2.0. The Diffusion-LM code keeps its original license.
