# Vesuvius Challenge — cross-scroll ink-detection harness

Reproducible inference and **cross-scroll** evaluation of public ink-detection models for the
[Vesuvius Challenge](https://scrollprize.org), built on the open ecosystem instead of re-implementing it.

**Status:** Phase 0b — first inference of `ink_9um` on a small crop of PHerc 0841 (Kaggle T4).

## Core rule: no contamination
A metric is reported as *cross-scroll* only when the evaluated scroll is provably absent from the
checkpoint's training data. The registry is [`configs/checkpoints.yaml`](configs/checkpoints.yaml), sourced
from the official training config (pinned commit). Notably, **every PHerc Paris 4 segment in
`ink-labels` is in `ink_9um`'s training set**, so Paris 4 is only an in-distribution control here.

## What is reused (not re-implemented)
| Component | Source | Used for |
|---|---|---|
| `vesuvius.ink_detection` (villa, MIT) | [ScrollPrize/villa](https://github.com/ScrollPrize/villa) @ `0e14cf48` | inference CLI, remote Zarr I/O |
| `ink_9um` weights (MIT) | [scrollprize/ink_9um](https://huggingface.co/scrollprize/ink_9um) @ `7109667e` | checkpoint under test |
| `ink-labels` dataset | HF bucket `scrollprize/datasets/ink` | surface volumes + ink / supervision / validation masks |

This repo only adds thin glue: crop selection/fetch (`scripts/fetch_crop.py`), experiment configs,
the Kaggle notebook and (Phase 0c) metrics and reports.

## Layout
```
configs/      one YAML per experiment + checkpoints.yaml (contamination registry)
scripts/      loose scripts (moved to src/ only once reused)
notebooks/    Kaggle notebooks
data/ outputs/ checkpoints/   git-ignored
```

## Reproduce exp000 (Kaggle)
1. New Kaggle notebook → *File → Import notebook* → `notebooks/00_kaggle_exp000_841w00_ink9um.ipynb`.
2. Accelerator **GPU T4**, Internet **on**, keep the notebook **private**.
3. Run the cells in order; the data-terms cell (`vesuvius.accept_terms --yes`) must be run by you.
4. Results land in `/kaggle/working/outputs/<experiment>/` (`crop.json`, `run.json`, `infer.log`, prediction TIFFs).

## Data provenance
- Segment: PHerc 0841 / `w00` from `ink-labels` (`https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/w00`),
  pyramid level 1 (~9.36 µm, hypothesis verified at fetch time), labels `*_v2`.
- Crop selection is deterministic and documented in the experiment config.

## Licenses
Code: MIT. Datasets used for training and derived checkpoints/pseudo-labels: CC BY-NC 4.0.
Data terms, EduceLab-Scrolls citation and the PHerc Paris 4 restriction: see [LICENSES.md](LICENSES.md).

## Links
[scrollprize.org](https://scrollprize.org) · [villa](https://github.com/ScrollPrize/villa) ·
[Hugging Face](https://huggingface.co/scrollprize) · Discord: _TBD_
