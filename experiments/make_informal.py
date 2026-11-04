"""Tạo bộ dev `data/train_informal/` mô phỏng phong cách câu hỏi Public Test.

Public Test phần lớn là câu văn nói: không dấu, viết tắt, gọi CLB bằng tên gián tiếp
("clb lap trinh cua hv", "clb game", "clb mm"). Tập train chỉ có 26 câu kiểu này, nên bộ dev
này chuyển các câu train còn lại sang cùng phong cách bằng quy tắc tất định (không gọi LLM),
giữ nguyên đáp án chuẩn để chấm được bằng `neorag run train_informal`.

Chạy: python experiments/make_informal.py
"""

from __future__ import annotations

import random
import re
import unicodedata

from neorag.data import DATA_DIR, read_jsonl, write_jsonl

SEED = 2026

# Tên CLB → cách gọi văn nói (trích từ câu hỏi Public Test / train không dấu).
CLUB_ALIASES = {
    r"CLB Lập Trình PTIT|ProPTIT|PROPTIT": [
        "clb lap trinh cua hv", "clb lap trinh ptit", "proptit", "clb code cua hv", "clb lt",
    ],
    r"CLB Multimedia PTIT|Multimedia PTIT|Multimedia|MPC": [
        "clb mm", "clb da phuong tien", "clb multimedia", "mpc",
    ],
    r"CLB Nhà Sáng Tạo Game PTIT(?: \(GCC\))?|Nhà Sáng Tạo Game|GCC": [
        "clb game", "clb lam game cua hv", "gcc",
    ],
}

# Cụm từ → viết tắt (áp dụng sau khi bỏ dấu, chữ thường).
ABBREV = [
    (r"\bcau lac bo\b", "clb"), (r"\bhoc vien\b", "hv"), (r"\bthanh vien\b", "tv"),
    (r"\bkhong\b", "ko"), (r"\bduoc\b", "dc"), (r"\bgi\b", "j"), (r"\bdang ky\b", "dk"),
    (r"\bem\b", "e"), (r"\bnhu the nao\b", "ntn"), (r"\bvay\b", "v"),
]


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def informalize(query: str, rng: random.Random) -> str:
    for pattern, aliases in CLUB_ALIASES.items():
        alias = rng.choice(aliases)
        query = re.sub(pattern, alias, query)
    q = strip_accents(query).lower()
    q = re.sub(r"\bạ\b|\ba\b(?=\s*\?)", "", q)
    for pattern, repl in ABBREV:
        q = re.sub(pattern, repl, q)
    q = re.sub(r"[?!.]+\s*$", "", q).strip()
    q = re.sub(r"\s+", " ", q)
    return q + rng.choice([" a", " v", "", " ạ"]).replace("ạ", "a")


def main() -> None:
    rng = random.Random(SEED)
    rows = read_jsonl(DATA_DIR / "train" / "questions.jsonl")
    out = []
    for row in rows:
        if row.get("type") == "không dấu/viết tắt":
            continue  # đã ở dạng văn nói
        new = dict(row)
        new["id"] = row["id"].replace("train-", "inf-")
        new["query"] = informalize(row["query"], rng)
        new["source_query"] = row["query"]
        out.append(new)
    dest = DATA_DIR / "train_informal" / "questions.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest, out)
    print(f"Đã ghi {len(out)} câu → {dest}")
    for r in out[:8]:
        print(f"- {r['source_query'][:70]}\n  → {r['query'][:90]}")


if __name__ == "__main__":
    main()
