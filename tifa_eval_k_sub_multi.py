"""
TIFA k-subset

- Executes the full pipeline N times (default 5) to capture any VQA randomness.
- Stores all runs in one CSV: correlations_vs_human_all_runs.csv.
- Each run uses a separate cache file: vqa_cache_run{r}.json (so results don't cross-contaminate).

Usage (example):
  python tifa_eval_k_sub_multi.py \
    --images_dir /path/to/annotated_images \
    --human_json /path/to/human_annotations.json \
    --qa_json /path/to/tifa_repo/tifa_v1.0/tifa_v1.0_question_answers.json \
    --out_dir /path/to/out \
    --k_values 4,8,12,16,24,32 \
    --random_seeds 0 \
    --num_runs 5
"""

import argparse, json, re, os, sys, csv, math, random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# ---------- dependency checks ----------
def _need(pkg):
    print(f"[Error] Missing package '{pkg}'. Please `pip install {pkg}`.", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception:
    _need("torch")
try:
    from PIL import Image
except Exception:
    _need("Pillow")
try:
    from tqdm.auto import tqdm
except Exception:
    _need("tqdm")
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except Exception:
    _need("transformers")
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    _need("sentence-transformers")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Paper-faithful TIFA k-sub evaluation (per-sample_id) — multi-run")
    ap.add_argument("--images_dir", required=True, help="Directory containing the annotated images")
    ap.add_argument("--human_json", required=True, help="Path to human_annotations.json")
    ap.add_argument("--qa_json", required=True, help="Path to tifa_v1.0_question_answers.json")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--model_id", default="Salesforce/blip2-flan-t5-xl", help="HF model id for BLIP-2")
    ap.add_argument("--sbert_id", default="sentence-transformers/all-mpnet-base-v2", help="SBERT model id")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Force device")
    ap.add_argument("--max_new_tokens", type=int, default=8, help="Max new tokens for BLIP-2")
    ap.add_argument("--k_values", default="4,8,12,16,24,32", help="Comma-separated ks")
    ap.add_argument("--random_seeds", default="0", help="Comma-separated seeds for random/stratified k")
    ap.add_argument("--num_runs", type=int, default=5, help="Number of independent full runs")
    ap.add_argument("--global_seed", type=int, default=0, help="Base seed; each run uses global_seed+run")
    ap.add_argument("--save_intermediates", action="store_true", help="Save per-question CSV, subset JSONs, etc.")
    return ap.parse_args()

# ---------- utils ----------
def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.-]", "", s)
    words = {
        "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
        "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
        "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
        "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20",
        "yeah":"yes","yep":"yes","nope":"no"
    }
    toks = [words.get(tok, tok) for tok in s.split()]
    return " ".join(toks)

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

# ---------- correlations (no SciPy) ----------
def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks

def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a -= a.mean()
    b -= b.mean()
    denom = a.std(ddof=1) * b.std(ddof=1)
    return float((a * b).sum() / ((len(a) - 1) * denom)) if denom > 0 else 0.0

def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata_avg_ties(np.asarray(a, dtype=float))
    rb = _rankdata_avg_ties(np.asarray(b, dtype=float))
    return pearsonr(ra, rb)

def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    concord = discord = ties_a = ties_b = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = int(a[i] > a[j]) - int(a[i] < a[j])
            db = int(b[i] > b[j]) - int(b[i] < b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1; continue
            if db == 0:
                ties_b += 1; continue
            if da == db:
                concord += 1
            else:
                discord += 1
    denom = math.sqrt((concord + discord + ties_a) * (concord + discord + ties_b))
    return float((concord - discord) / denom) if denom > 0 else 0.0

# ---------- one full run ----------
def run_once(run_idx: int, args, images_dir: Path, out_dir: Path,
             samples, qas_by_tid, samples_by_tid, device: str) -> List[Dict]:
    # Seeds per run (for any stochastic ops you may enable later)
    seed = args.global_seed + run_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # models
    print(f"[Run {run_idx}] Loading BLIP-2: {args.model_id}")
    processor = Blip2Processor.from_pretrained(args.model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print(f"[Run {run_idx}] Loading SBERT: {args.sbert_id}")
    sbert = SentenceTransformer(args.sbert_id)

    # per-run cache file so we recompute each run (helpful if VQA introduces randomness)
    cache_path = out_dir / f"vqa_cache_run{run_idx}.json"
    vqa_cache = load_json(cache_path) if cache_path.exists() else {}
    print(f"[Run {run_idx}] VQA cache entries: {len(vqa_cache)}")

    def vqa_short_answer(img_path: str, question: str, max_new_tokens: int = 8) -> str:
        # include run in key to avoid cross-run reuse
        prompt = question + " Answer with a single word or short phrase."
        key = f"run={run_idx}||{img_path}||{prompt}"
        if key in vqa_cache:
            return vqa_cache[key]
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        ans = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        vqa_cache[key] = ans
        if len(vqa_cache) % 200 == 0:
            save_json(cache_path, vqa_cache)
        return ans

    def map_to_choice_sbert(free_form_answer: str, choices: List[str]) -> Tuple[str, float]:
        if not choices:
            return free_form_answer, 0.0
        cands = [c.strip() for c in choices]
        emb_ans = sbert.encode([norm_text(free_form_answer)], normalize_embeddings=True)
        emb_choices = sbert.encode([norm_text(c) for c in cands], normalize_embeddings=True)
        sims = sbert_util.cos_sim(emb_ans, emb_choices).cpu().numpy().ravel()
        if sims.size == 0:
            return cands[0], 0.0
        best = int(np.argmax(sims))
        return cands[best], float(sims[best])

    # ----- full pass per-sample -----
    per_q = []
    print(f"[Run {run_idx}] Running VQA per sample...")
    for s in tqdm(samples):
        tid = s["text_id"]
        img_path = str(images_dir / s["image_fname"])
        qs = qas_by_tid.get(tid, [])
        if not qs:
            continue
        for q in qs:
            free = vqa_short_answer(img_path, q["question"], max_new_tokens=args.max_new_tokens)
            pred_choice, conf = map_to_choice_sbert(free, q.get("choices", []))
            gold = q["answer"]
            per_q.append({
                "sample_id": s["sample_id"],
                "text_id":   tid,
                "question":  q["question"],
                "gold":      gold,
                "pred":      pred_choice,
                "is_correct": int(norm_text(pred_choice) == norm_text(gold)),
                "conf":      float(conf),
                "category":  q.get("category", q.get("q_type", "uncat")),
            })
    save_json(cache_path, vqa_cache)

    if args.save_intermediates:
        write_csv(out_dir / f"per_question_full_run{run_idx}.csv", per_q,
                  ["sample_id","text_id","question","gold","pred","is_correct","conf","category"])

    # per-sample TIFA
    scores_by_sample: Dict[str, List[float]] = {}
    for r in per_q: scores_by_sample.setdefault(r["sample_id"], []).append(float(r["is_correct"]))
    per_sample = [{"sample_id": sid, "tifa_all": float(np.mean(v)) if v else np.nan}
                  for sid, v in scores_by_sample.items()]
    if args.save_intermediates:
        write_csv(out_dir / f"per_image_full_run{run_idx}.csv", per_sample, ["sample_id","tifa_all"])

    # confidence + correctness lookups
    conf_map = {(r["sample_id"], r["question"]): float(r["conf"]) for r in per_q}
    correct_map = {(r["sample_id"], r["question"]): int(r["is_correct"]) for r in per_q}

    def qa_cat(q): return q.get("category", q.get("q_type", "uncat"))

    # subset builders per text_id
    def build_random_k_for_tid(tid: str, k: int, rng: random.Random) -> List[dict]:
        qs = list(qas_by_tid.get(tid, []))
        return rng.sample(qs, min(k, len(qs)))

    def build_stratified_k_for_tid(tid: str, k: int, rng: random.Random) -> List[dict]:
        qs = qas_by_tid.get(tid, [])
        total = len(qs)
        if total == 0:
            return []
        cats: Dict[str, List[dict]] = {}
        for q in qs:
            cats.setdefault(qa_cat(q), []).append(q)
        alloc = {c: max(1, round(k * len(v) / total)) for c, v in cats.items()}
        while sum(alloc.values()) > min(k, total):
            cmax = max(alloc, key=lambda c: alloc[c])
            if alloc[cmax] > 1: alloc[cmax] -= 1
            else: break
        pick = []
        for c, arr in cats.items():
            arr = arr[:]; rng.shuffle(arr)
            pick += arr[:alloc[c]]
        return pick[:min(k, total)]

    def build_conf_k_for_sample(tid: str, k: int, sample_id: str) -> List[dict]:
        qs = qas_by_tid.get(tid, [])
        scored = [(conf_map.get((sample_id, q["question"]), 0.0), q) for q in qs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in scored[:min(k, len(scored))]]

    def eval_subset_for_samples(sub_qs: List[dict], sample_ids: List[str]) -> Dict[str, float]:
        acc = {}
        for sid in sample_ids:
            vals = []
            for q in sub_qs:
                key = (sid, q["question"])
                if key in correct_map:
                    vals.append(float(correct_map[key]))
            acc[sid] = float(np.mean(vals)) if vals else np.nan
        return acc

    # base table per sample
    tifa_table: Dict[str, Dict[str, float]] = {row["sample_id"]: {"tifa_all": row["tifa_all"]} for row in per_sample}

    # k loops
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    seeds    = [int(x) for x in args.random_seeds.split(",") if x.strip()]

    for k in k_values:
        for seed_local in seeds:
            rng = random.Random(seed_local + seed * 100003)  # mix run seed for independence
            name_r = f"tifa_random_k{k}_seed{seed_local}"
            name_s = f"tifa_strat_k{k}_seed{seed_local}"
            for tid, sids in samples_by_tid.items():
                sub_r = build_random_k_for_tid(tid, k, rng)
                sub_s = build_stratified_k_for_tid(tid, k, rng)
                if args.save_intermediates:
                    if sub_r: save_json(out_dir / f"qas_random_k{k}_seed{seed_local}_tid{tid}_run{run_idx}.json", sub_r)
                    if sub_s: save_json(out_dir / f"qas_strat_k{k}_seed{seed_local}_tid{tid}_run{run_idx}.json",  sub_s)
                er = eval_subset_for_samples(sub_r, sids) if sub_r else {sid: np.nan for sid in sids}
                es = eval_subset_for_samples(sub_s, sids) if sub_s else {sid: np.nan for sid in sids}
                for sid, v in er.items(): tifa_table.setdefault(sid, {})[name_r] = v
                for sid, v in es.items(): tifa_table.setdefault(sid, {})[name_s] = v

        name_c = f"tifa_conf_k{k}"
        for tid, sids in samples_by_tid.items():
            for sid in sids:
                sub_c = build_conf_k_for_sample(tid, k, sid)
                ec = eval_subset_for_samples(sub_c, [sid]) if sub_c else {sid: np.nan}
                tifa_table.setdefault(sid, {})[name_c] = ec[sid]

    # correlate vs human per metric
    human_avg_by_sample = {s["sample_id"]: s["human_avg"] for s in samples}
    all_sids = sorted(tifa_table.keys())
    metrics = sorted({m for d in tifa_table.values() for m in d.keys()})
    results = []
    for m in metrics:
        a = []; b = []
        for sid in all_sids:
            hv = human_avg_by_sample.get(sid, np.nan)
            mv = tifa_table.get(sid, {}).get(m, np.nan)
            if not (np.isnan(hv) or np.isnan(mv)):
                a.append(hv); b.append(mv)
        if not b:
            results.append({"run": run_idx, "metric": m, "n": 0, "spearman": np.nan, "kendall": np.nan, "pearson": np.nan})
            continue
        a = np.array(a, dtype=float); b = np.array(b, dtype=float)
        results.append({
            "run": run_idx, "metric": m, "n": int(len(b)),
            "spearman": float(spearmanr(a, b)),
            "kendall":  float(kendall_tau_b(a, b)),
            "pearson":  float(pearsonr(a, b)),
        })

    # optional wide per-sample dump per run
    if args.save_intermediates:
        metric_cols = ["tifa_all"] + [r["metric"] for r in results if r["metric"] != "tifa_all"]
        per_sample_rows = []
        for sid in all_sids:
            row = {"sample_id": sid, "human_avg": human_avg_by_sample.get(sid)}
            row.update({m: tifa_table.get(sid, {}).get(m, None) for m in metric_cols})
            per_sample_rows.append(row)
        write_csv(out_dir / f"per_image_all_conditions_run{run_idx}.csv",
                  per_sample_rows, ["sample_id","human_avg"] + metric_cols)

    return results

# ---------- main ----------
def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # device
    device = args.device if args.device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # static inputs (shared across runs)
    human = load_json(Path(args.human_json))   # sample_id -> {text_id, image_path, human_avg, ...}
    qa_list = load_json(Path(args.qa_json))    # list of QAs

    # samples
    samples = []
    if isinstance(human, dict):
        for sample_id, rec in human.items():
            samples.append({
                "sample_id":   sample_id,
                "text_id":     rec.get("text_id"),
                "image_fname": rec.get("image_path"),
                "human_avg":   rec.get("human_avg"),
            })
    else:
        for rec in human:
            samples.append({
                "sample_id":   rec.get("sample_id"),
                "text_id":     rec.get("text_id"),
                "image_fname": rec.get("image_path"),
                "human_avg":   rec.get("human_avg"),
            })

    present_imgs = {p.name for p in images_dir.glob("*")}
    samples = [s for s in samples if s["image_fname"] in present_imgs and s["text_id"]]

    # QAs grouped by tid
    qas_by_tid: Dict[str, List[dict]] = {}
    for q in qa_list:
        tid = q.get("id")
        if tid:
            qas_by_tid.setdefault(tid, []).append(q)

    # samples grouped by tid
    samples_by_tid: Dict[str, List[str]] = {}
    for s in samples:
        samples_by_tid.setdefault(s["text_id"], []).append(s["sample_id"])

    print(f"[Info] Samples: {len(samples)} | Unique text_ids: {len(set(s['text_id'] for s in samples))}")
    print(f"[Info] Running {args.num_runs} full runs...")

    all_results: List[Dict] = []
    for r in range(args.num_runs):
        print(f"\n========== RUN {r} ==========")
        res = run_once(r, args, images_dir, out_dir, samples, qas_by_tid, samples_by_tid, device)
        all_results.extend(res)

    # write combined CSV
    write_csv(out_dir / "correlations_vs_human_all_runs.csv",
              all_results, ["run","metric","n","spearman","kendall","pearson"])

    print("\n[Done]")
    print("  -", out_dir / "correlations_vs_human_all_runs.csv")
    if args.save_intermediates:
        print("  - per_question_full_run{r}.csv, per_image_full_run{r}.csv, per_image_all_conditions_run{r}.csv")

if __name__ == "__main__":
    main()

