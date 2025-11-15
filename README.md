# TIFA k-Subset (Paper-Faithful) — Reproduction & Speed-ups

This repo reproduces **TIFA** (Text-to-Image Faithfulness) on the official 800 human-annotated prompts and adds **k-subset question selection** to reduce VQA compute while keeping correlation with human judgments high.

> **Note:** The pipeline is implemented in a **single script**:
>
> * **`tifa_eval_k_sub_multi.py`** — runs the **entire** evaluation **N times** (default 5) to capture any VQA randomness, and writes all correlations into one CSV.

---

## Quick Links (Upstream)

* Official TIFA repo (paper, data, QAs): [https://github.com/Yushi-Hu/tifa](https://github.com/Yushi-Hu/tifa)
* BLIP-2 (Flan-T5-XL) model card: [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
* Sentence-Transformers `all-mpnet-base-v2`: [https://huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

---

## What the Script Does

`tifa_eval_k_sub_multi.py`:

* Loads **human annotations** (per **sample_id**) where multiple images can share the same `text_id` (different generators).
* Loads official **TIFA v1.0 QAs** (`tifa_v1.0/tifa_v1.0_question_answers.json`).
* Runs **BLIP-2** VQA with a short-answer prompt, then maps free-form answers to the provided **MC choices** via **SBERT cosine similarity**.
* Computes **per-sample TIFA** = mean MC correctness across that sample’s questions.
* Builds **k-subsets** per `text_id` (**random-k**, **stratified-k**) and per **sample** (**confidence-k** using SBERT similarity).
* Re-scores using cached per-question correctness (no re-VQA) and computes **Spearman/Kendall/Pearson** correlations vs. **human_avg**.
* Repeats the **whole pipeline** for `--num_runs` and aggregates results.

**Outputs**

* `correlations_vs_human_all_runs.csv` — one row per **(run, metric)** with `spearman`, `kendall`, `pearson`, and `n`.
* (optional with `--save_intermediates`) per-run CSVs and subset dumps.

---

## Requirements

Install deps:

```bash
pip install "torch>=2.1" "transformers>=4.43" "accelerate>=0.33" pillow tqdm sentence-transformers
```

**`requirements.txt` (minimal):**

```
torch>=2.1
transformers>=4.43
accelerate>=0.33
pillow
tqdm
sentence-transformers
```

---

## Data You Need

1. **Official TIFA QAs (v1.0)**
   From the TIFA repo: `tifa_v1.0/tifa_v1.0_question_answers.json`.

2. **Human annotations + 800 images**

   * A consolidated JSON mapping **sample_id → { text_id, image_path, human_avg, … }**.
     (Evaluation is **per `sample_id`** so different generators for the same `text_id` are scored separately.)
   * A directory containing the **800 image files**; filenames must match the `image_path` in the JSON.

> **Keying:** QAs are grouped by **text_id**; multiple images (different generators) can share a `text_id`. The script scores each **sample_id** against its `text_id`’s QAs.

---

## How the Score Is Computed (Paper-Faithful Shape)

For each **sample (image)** with a given `text_id`:

1. Run **BLIP-2** to produce a **short free-form** answer per question
   (prompt suffix: “**Answer with a single word or short phrase.**”).
2. Map the free-form answer to one of the **provided MC choices** via **SBERT** cosine similarity.
3. Mark **correct** if the mapped choice equals the **gold** choice; else **incorrect**.

Per-sample **TIFA** = mean(correctness across its questions).
We then compute **Spearman / Kendall / Pearson** correlations between per-sample TIFA and **human_avg** over all samples.

> ⚠️ **Exact matching note:** The original paper’s `tifascore` applies a reference normalization/matching (number words, yes/no variants, etc.) before exact matching.
> This script uses a compact normalizer + SBERT mapping, which is behaviorally close but not byte-identical.
> To **match the paper exactly**, swap the per-question “is_correct” with the official `tifascore` API.

---

## k-Subset Selection (Compute Savings)

Let each prompt have (Q) questions (empirically **~6.3** on average):

* **random-k:** uniform sample of `k` per `text_id`.
* **stratified-k:** proportional by question category (`category`/`q_type`) with rounding and a small fix-up to hit `k`.
* **confidence-k:** per-sample ranking by SBERT similarity; take top-`k` for that sample.

**Rough VQA savings:** (\text{savings} \approx 1 - \frac{\min(Q,k)}{Q}).

* `k = 4` → ≈ **36.5%** fewer VQA calls (with (Q \approx 6.3))
* `k = 8` → often **no savings** (since many prompts have ≤ 8 Qs)

---

## Usage

### Run N Full Trials (default N=5)

```bash
python tifa_eval_k_sub_multi.py \
  --images_dir /path/to/800_imgs \
  --human_json /path/to/human_annotations.json \
  --qa_json /path/to/tifa_repo/tifa_v1.0/tifa_v1.0_question_answers.json \
  --out_dir /path/to/out_multi \
  --k_values 4,8,12,16,24,32 \
  --random_seeds 0,1,2 \
  --num_runs 5 \
  --global_seed 0 \
  --device auto \
  --save_intermediates
```

**Main output**

* `correlations_vs_human_all_runs.csv`

**Optional (with `--save_intermediates`)**

* Per-run VQA caches: `vqa_cache_run{r}.json`
* Per-run question-level: `per_question_full_run{r}.csv`
* Per-run wide table: `per_image_all_conditions_run{r}.csv`
* Chosen subsets per `tid`: `qas_*_tid{tid}_run{r}.json`

---

## Reproducing the Results in This Repo

1. Clone the **official TIFA** repo and locate
   `tifa_v1.0/tifa_v1.0_question_answers.json`.
2. Prepare your **human_annotations.json** with the **per-sample_id** schema and place the **800 images** into a directory. (or use files under human_annotations)
3. Create a Python venv and install **Requirements**.
4. Run `tifa_eval_k_sub_multi.py` as shown above.
5. Inspect `correlations_vs_human_all_runs.csv` (and per-run CSVs if saved).

---

## Differences vs. the Original Paper (and How to Close the Gap)

* **Normalization / Matching**
  The paper’s reference scorer (`tifascore`) applies canonical answer normalization (number words, yes/no variants, etc.) before exact matching.
  **This script** uses a compact normalizer + **SBERT choice mapping** for robustness. Close in spirit, **not** byte-identical.
  **To match exactly:** replace the correctness step with the official `tifascore` API.

* **VQA Backbone**
  We default to **BLIP-2 (Flan-T5-XL)**. If you want the exact leaderboard numbers, use the specific VQA model(s) reported by the paper/official repo.

---

## FAQ

**Why can adding more questions lower correlation with humans?**
TIFA is a **mean**. Appending many **easy/low-signal** questions pulls per-sample scores toward the mean, reducing the influence of **hard/high-signal** questions that better track human judgment (rank-based correlations drop).

**Where do I find the final numbers?**
All runs are in **`correlations_vs_human_all_runs.csv`**.

---

## Citation

If you use this code or the human annotations, please cite the original TIFA work:

* **TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering.**
  GitHub: [https://github.com/Yushi-Hu/tifa](https://github.com/Yushi-Hu/tifa)

Models:

* BLIP-2 (Flan-T5-XL): [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
* Sentence-Transformers (all-mpnet-base-v2): [https://huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

---

## Acknowledgements

Thanks to the TIFA authors for releasing the benchmark, QAs, and human annotations.
