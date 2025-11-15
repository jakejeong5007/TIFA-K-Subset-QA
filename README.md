# TIFA k-Subset (Paper-Faithful) — Reproduction & Speed-ups

This repo reproduces **TIFA** (Text-to-Image Faithfulness) on the official 800 human-annotated prompts and adds **k-subset question selection** to reduce VQA compute while keeping correlation with human judgments high.

It includes:

- **`tifa_eval_k_sub.py`** — single-run, paper-faithful evaluation + k-subset selection  
- **`tifa_eval_k_sub_multi.py`** — repeats the full pipeline **N** times (default 5) to capture VQA randomness

**k-subset strategies:** `random-k`, `stratified-k`, and `confidence-k` (per sample)

---

## Quick Links (Upstream)

- Official TIFA repo (paper, data, QAs): https://github.com/Yushi-Hu/tifa  
- BLIP-2 (Flan-T5-XL) model card: https://huggingface.co/Salesforce/blip2-flan-t5-xl  
- Sentence-Transformers `all-mpnet-base-v2`: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

---

## What’s Here

- **`tifa_eval_k_sub.py`**
  - Inputs: human annotations JSON, the 800 images, official `tifa_v1.0/tifa_v1.0_question_answers.json`
  - Outputs:
    - `correlations_vs_human.csv` — Spearman/Kendall/Pearson vs human averages
    - `per_image_all_conditions.csv` — wide per-sample table of all metrics
    - (optional) `per_question_full.csv`, `per_image_full.csv`, `qas_*_tid.json` with `--save_intermediates`

- **`tifa_eval_k_sub_multi.py`**
  - Runs the **entire** pipeline multiple times
  - Outputs: `correlations_vs_human_all_runs.csv` (rows = run × metric)

---

## Requirements

Create/activate a clean environment (example for BU SCC; adjust as needed):

```bash
python -m venv ~/venvs/tifa
source ~/venvs/tifa/bin/activate
pip install --upgrade pip
