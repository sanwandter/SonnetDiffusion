# SonnetDiffusion

Code, data splits, model checkpoint and raw results for **"SonnetDiffusion: Untrained Analytic
Control for Formal Constraints in Diffusion Poetry Generation."**

Anonymized repository for double-blind review. Please do not deanonymize.

A continuous-latent diffusion language model (Diffusion-LM family, 87.5M parameters) fine-tuned on
Spanish sonnets, whose latent is corrected at almost every denoising step by the analytic gradient of
six closed-form differentiable penalties. No component of the controller is trained.

---

## Layout

| Path | Contents |
|---|---|
| `sonnetdiffusion/` | The method. `controllers_es.py` (the six penalties and the control loop), `rhyme_es.py` (phonetic suffix rule, syllable counting), `eval_sonnets_es.py` (all four metrics), `vocab_es.py`, `pipeline.py`, `model_utils.py`, `encoder.py`, `interaction.py` |
| `experiments/` | One script per table/figure: `ablation_seeded.py`, `best_of_n.py`, `k_sweep.py`, `lambda_sweep.py`, `corpus_meter_stats.py`, `score_llm_baseline.py`, plus the shared `harness.py` and `build_tables.py` |
| `data/` | Raw results behind every number in the paper, as written by the scripts above. `data/llm_baseline/` holds the prompted Claude Haiku 4.5 outputs and their scores; `data/samples/` holds generated sonnets |
| `datasets/` | The 8,648 / 480 / 480 train/valid/test splits and the two source corpora |
| `checkpoints/` | The 50k EMA checkpoint used throughout the paper, with its vocabulary and embedding matrix |
| `improved-diffusion/` | Vendored Diffusion-LM training/sampling code, lightly modified (`logger.py`, `resample.py`, `text_datasets.py`) |
| `tests/` | Harness unit tests |

## Setup

```bash
git lfs install && git clone <this repo>      # the checkpoint is stored via Git LFS
pip install -r requirements.txt
```

Results in the paper were produced on a single RTX 3090 (24 GB).

## Reproducing the tables

```bash
cd experiments
python ablation_seeded.py     # Table 4 (per-controller ablation) and the Table 1 lower block
python best_of_n.py           # Table 6, Figure 2 (oracle best-of-N frontier)
python k_sweep.py             # Table 3 (inner-step budget K)
python lambda_sweep.py        # Table 5 (weight sweeps)
python corpus_meter_stats.py  # Table 1 human-corpus row
python build_tables.py        # regenerates data/paper_tables.json from the above
```

Every configuration runs three seeds (`102, 202, 302`, set in `harness.py`) at 30 sonnets per seed.
Reported intervals are the mean ± the half-width of a 95% Student-*t* interval across the three seed
means (2 degrees of freedom).

## Operating point

The headline configuration, from `harness.py`:

```python
FULL = dict(lambda_fmt=0.05, lambda_rhyme=0.5, lambda_rep=0.2, lambda_syll=0.05,
            lambda_concept=0.15, lambda_emotion=0.15, K=5)
```

with `coef=5e-4` (the posterior anchor), `step_size=0.03`, `t_stop=10`, `temperature=0.8`, and the
per-step noise scale floored at `0.1` — all in `sonnetdiffusion/controllers_es.py`.

## Data and model licenses

- **DISCO corpus** — CC-BY 4.0.
- **Golden Age sonnets corpus** — metrical annotations under CC-BY-NC 4.0.
- **SigLIP 2** — Apache 2.0.
- **RoBERTuito** — MIT.
- Claude Haiku 4.5 was accessed through a commercial API under its provider's terms.
- **improved-diffusion / Diffusion-LM** — vendored under its original license; see the directory.
