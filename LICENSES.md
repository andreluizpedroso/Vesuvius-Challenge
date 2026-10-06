# Licenses and data-use terms

| What | License / terms |
|---|---|
| Code in this repository | MIT (see `LICENSE`) |
| Any dataset created or used here for training, and derived checkpoints / pseudo-labels | CC BY-NC 4.0 |
| Vesuvius Challenge data (general) | CC BY-NC 4.0 unless noted otherwise (https://scrollprize.org/data) |
| `ink-labels` dataset (HF bucket `scrollprize/datasets/ink`) | See https://dl.ash2txt.org/LICENSE.txt |
| Scrolls 1–4 and Fragments 1–6 (EduceLab-Scrolls) | EduceLab-Scrolls terms (below) |
| Upstream code: `ScrollPrize/villa` (incl. `vesuvius`) | MIT |
| Weights: `scrollprize/ink_9um` | MIT (model card) |

## EduceLab-Scrolls terms (Scrolls 1–4, Fragments 1–6)

These apply to **PHerc. Paris 4 (= Scroll 1)**. Summary of the terms printed by
`vesuvius.accept_terms` (read the full text there before use):

- No redistribution of the data without written approval of Vesuvius Challenge.
- No public revelation of hidden text (or associated code) outside the Discord
  without written approval.
- Publications must cite EduceLab-Scrolls: Parsons, S., Parker, C. S., Chapman, C.,
  Hayashida, M., & Seales, W. B. (2023). *EduceLab-Scrolls: Verifiable Recovery of
  Text from Herculaneum Papyri using X-ray CT.* arXiv:2304.02084.

## PHerc. Paris 4: publication restriction

Academic publication rights for PHerc. Paris 4 are reserved to designated teams
until 2027-06-25. Analysis is allowed; this project makes **no academic claims**
about Paris 4 content.

## Practical rules for this repository (public)

- Never commit data, predictions, renders, ink images or weights (enforced by `.gitignore`).
- Only aggregate numeric results (metrics tables) are committed.
- Data terms are accepted by the repository owner personally
  (`vesuvius.accept_terms --yes`); no script in this repo runs that command.
