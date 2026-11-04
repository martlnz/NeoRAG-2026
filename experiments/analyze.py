"""Phân tích kết quả chấm train theo loại câu hỏi (đọc file có sẵn, không gọi API).

Chạy sau `neorag run <split> --pipeline ...`:
    python experiments/analyze.py              # train
    python experiments/analyze.py train_informal
In ra bảng metric theo `type`, độ dài câu trả lời, các câu trượt retrieval và các câu noise cao.
"""

from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from neorag.data import read_jsonl

METRICS = [
    "recall@5", "mrr@5", "ndcg@5", "string_presence", "rouge_l",
    "groundedness", "response_relevancy", "noise_sensitivity", "abstention",
]


def avg(rows: list[dict], metric: str) -> str:
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    return f"{st.mean(vals):.3f}" if vals else "—"


def main(split: str = "train", root: Path = Path(".")) -> None:
    gold = {q["id"]: q for q in read_jsonl(root / f"data/{split}/questions.jsonl")}
    pred = {p["id"]: p for p in read_jsonl(root / f"outputs/{split}_pred.jsonl")}
    rows = read_jsonl(root / f"outputs/eval_{split}_pred/per_query.jsonl")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[gold[r["id"]]["type"]].append(r)

    print("| Loại câu hỏi | n | " + " | ".join(METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for t, rs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"| {t} | {len(rs)} | " + " | ".join(avg(rs, m) for m in METRICS) + " |")

    lens = [len(p["answer"]) for p in pred.values()]
    glens = [len(q["answer"]) for q in gold.values()]
    lat = [p.get("latency_ms") or 0 for p in pred.values()]
    print(
        f"\nĐộ dài câu trả lời: median {st.median(lens):.0f}, max {max(lens)} "
        f"| đáp án chuẩn: median {st.median(glens):.0f}"
    )
    print(f"Latency ms: median {st.median(lat):.0f}, max {max(lat):.0f}")

    miss = [
        r for r in rows
        if r.get("recall@5") is not None and r["recall@5"] < 1 and gold[r["id"]]["relevant_ids"]
    ]
    print(f"\n## Retrieval miss (recall@5 < 1): {len(miss)}")
    for r in miss:
        g = gold[r["id"]]
        got = pred[r["id"]]["retrieved_ids"][:5]
        print(f"- {r['id']} [{g['type']}] rel={g['relevant_ids']} got={got} :: {g['query'][:90]}")

    noisy = [r for r in rows if r.get("noise_sensitivity") is not None]
    noisy.sort(key=lambda r: (-r["noise_sensitivity"], r.get("string_presence") or 0))
    print("\n## Noise cao / String Presence thấp (8 câu)")
    for r in noisy[:8]:
        g = gold[r["id"]]
        answer = pred[r["id"]]["answer"][:220].replace("\n", " ")
        print(
            f"- {r['id']} [{g['type']}] noise={r['noise_sensitivity']:.2f} "
            f"sp={r.get('string_presence')} :: Q: {g['query'][:80]}\n"
            f"    GOLD: {g['answer'][:160]}\n    PRED: {answer}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
