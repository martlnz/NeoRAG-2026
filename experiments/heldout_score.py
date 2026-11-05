"""Điểm benchmark sau khi LOẠI các câu dùng làm few-shot (và bản informal của chúng).

Few-shot lấy từ train → chấm lại trên chính các câu đó sẽ thổi phồng điểm. Script tính lại
retrieval/generation/benchmark từ `outputs/eval_<split>_pred/per_query.jsonl`, bỏ các id few-shot.

Chạy: python experiments/heldout_score.py train train_informal
"""

from __future__ import annotations

import statistics as st
import sys

from neorag.config import load_config
from neorag.data import read_jsonl
from neorag.pipelines.pipeline import FEWSHOT_IDS


def score(rows: list[dict], cfg: dict) -> tuple[float, float, float]:
    s = cfg["score"]
    k = s["rank_k"]

    def mean(metric: str) -> float | None:
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        return st.mean(vals) if vals else None

    retrieval = st.mean(mean(f"{m}@{k}") for m in s["retrieval_metrics"])
    parts = []
    for m in s["generation_metrics"]:
        v = mean(m)
        parts.append(1 - v if m == "noise_sensitivity" else v)
    if s.get("abstention_in_generation") and mean("abstention") is not None:
        parts.append(mean("abstention"))
    generation = st.mean(parts)
    return retrieval, generation, s["retrieval_weight"] * retrieval + s["generation_weight"] * generation


def main(splits: list[str]) -> None:
    cfg = load_config()
    excluded = {i.split("-")[1] for i in FEWSHOT_IDS}  # số thứ tự: train-0085 ↔ inf-0085
    print("| Bộ | n | Retrieval | Generation | Benchmark |")
    print("|---|---|---|---|---|")
    for split in splits:
        rows = read_jsonl(f"outputs/eval_{split}_pred/per_query.jsonl")
        kept = [r for r in rows if r["id"].split("-")[1] not in excluded]
        for label, rs in ((f"{split} (đủ)", rows), (f"{split} (loại few-shot)", kept)):
            r, g, b = score(rs, cfg)
            print(f"| {label} | {len(rs)} | {r:.4f} | {g:.4f} | **{b:.4f}** |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "train_informal"])
