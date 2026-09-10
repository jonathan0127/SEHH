#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from typing import List, Optional, Dict, Set
import re

PREFERRED_ORDER = [
    "DE_b1", "DE_b2", "DE_r1", "DE_r2",
    "PSO", "LSHADE", "GA",
    "Hybrid1", "Hybrid2", "Hybrid3",
    # 可加上 "ESHHA" 若出現
]

DISPLAY_ALIAS: Dict[str, str] = {
    # 這裡可對應顯示名稱（若檔名不想直接露出）
    # 預設用原鍵即可
}

def detect_available_algs(result_dir: str, dim: int) -> List[str]:
    """掃描 result_dir 下符合 *_<dim>_*.txt 的檔名，收集實際演算法鍵。"""
    try:
        names = set(os.listdir(result_dir))
    except Exception:
        return []
    pat = re.compile(r"^(\d+)_%d_(.+)\.txt$" % dim)
    algs: Set[str] = set()
    for fn in names:
        m = pat.match(fn)
        if not m:
            continue
        alg = m.group(2)
        if alg:
            algs.add(alg)
    # 依偏好順序排序，未列入偏好者按字母序附加
    ordered: List[str] = [a for a in PREFERRED_ORDER if a in algs]
    leftovers = sorted(list(algs - set(ordered)))
    return ordered + leftovers

def detect_available_funcs(result_dir: str, dim: int) -> List[int]:
    """掃描 result 目錄下實際存在的函數索引 fidx（對應 <fidx>_<dim>_*.txt）。"""
    try:
        names = set(os.listdir(result_dir))
    except Exception:
        return []
    pat = re.compile(r"^(\d+)_%d_.+\.txt$" % dim)
    funcs: Set[int] = set()
    for fn in names:
        m = pat.match(fn)
        if not m:
            continue
        try:
            funcs.add(int(m.group(1)))
        except Exception:
            continue
    return sorted(funcs)


def read_final_value(file_path: str) -> Optional[float]:
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # 從檔尾往前找最後一個可解析的浮點數（不強制兩欄格式）
        for line in reversed(lines):
            parts = line.split()
            # 優先嘗試最後一欄
            if parts:
                try:
                    return float(parts[-1])
                except ValueError:
                    # fallback：用正則找行內第一個浮點數
                    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if m:
                        try:
                            return float(m.group(0))
                        except Exception:
                            pass
        return None
    except Exception:
        return None


def fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.2e}"


def is_close(a: float, b: float, rel: float = 1e-12, abs_tol: float = 1e-9) -> bool:
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)


def build_table(dim: int, result_dir: str, funcs: List[int]) -> str:
    columns = detect_available_algs(result_dir, dim)
    header = ["Func."] + columns
    lines = [
        "|" + " | ".join(header) + "|",
        "|" + "|".join(["-" * len(h) for h in header]) + "|",
    ]

    # 將功能索引 fidx 轉換成實際檔名前綴 index：
    # F1(10D)->1, F1(20D)->2, F2(10D)->3, F2(20D)->4, ...
    def file_index_for(fidx: int, dim: int) -> int:
        return (fidx - 1) * 2 + (1 if dim == 10 else 2)

    for fidx in funcs:
        # 先收集所有可用數值以求最小值
        values: Dict[str, Optional[float]] = {}
        idx_prefix = file_index_for(fidx, dim)
        for col in columns:
            key = col  # 直接用偵測到的鍵當檔名鍵
            file_name = f"{idx_prefix}_{dim}_{key}.txt"
            file_path = os.path.join(result_dir, file_name)
            values[col] = read_final_value(file_path)

        numeric_vals = [v for k, v in values.items() if v is not None]
        min_val = min(numeric_vals) if numeric_vals else None

        row = [f"F{fidx}(D{dim})"]
        for col in columns:
            v = values.get(col)
            label = DISPLAY_ALIAS.get(col, col)
            s = fmt(v)
            if (v is not None) and (min_val is not None) and is_close(v, min_val):
                s = f"**{s}**"
            row.append(s)
        lines.append("|" + " | ".join(row) + "|")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, choices=[10, 20], help="dimension: 10 or 20")
    parser.add_argument("--year", type=int, choices=[2020, 2021, 2022], default=2022, help="CEC benchmark year: 2020/2021/2022")
    parser.add_argument("--result-dir", type=str, default="result", help="result folder")
    parser.add_argument("--only-existing", action="store_true", help="only list functions that have files for the given dim")
    parser.add_argument("--funcs", nargs="*", type=str, default=None, help="function indices or ranges like 1-10")
    parser.add_argument("--output", type=str, default=None, help="write markdown to file path; if omitted, print to stdout")
    args = parser.parse_args()

    # 偵測此維度實際存在的檔名索引（如 1,2,3,... 對應 1_10_*.txt 等），再映射為函數索引 fidx
    available_file_indices = detect_available_funcs(args.result_dir, args.dim)
    max_funcs_by_year = {2020: 10, 2021: 10, 2022: 12}
    max_f = max_funcs_by_year.get(args.year, 12)
    # 將檔名索引映射為函數索引：fidx = ((idx-1)//2) + 1，且需符合該維度的奇偶性（10D=奇；20D=偶）
    def to_fidx(idx: int) -> int:
        return ((idx - 1) // 2) + 1
    parity = 1 if args.dim == 10 else 0
    existing_fidx = sorted({to_fidx(i) for i in available_file_indices if (i % 2) == parity})
    # 年份過濾在 fidx 上進行
    existing_fidx = [fi for fi in existing_fidx if 1 <= fi <= max_f]

    # parse funcs tokens, support ranges like 1-10；
    # 若未提供：預設列出 1..max_f（即使沒有檔案也會留空），
    # 若加上 --only-existing：則使用 available_funcs 清單。
    toks: Optional[List[str]] = args.funcs
    if toks:
        selected: List[int] = []
        for tk in toks:
            tk = tk.strip()
            if not tk:
                continue
            if "-" in tk:
                try:
                    a, b = tk.split("-", 1)
                    a = int(a)
                    b = int(b)
                    if a <= b:
                        selected.extend(list(range(a, b + 1)))
                    else:
                        selected.extend(list(range(b, a + 1)))
                except Exception:
                    continue
            else:
                try:
                    selected.append(int(tk))
                except Exception:
                    continue
        # 年份過濾；若 --only-existing 才與存在清單取交集
        selected = [fi for fi in selected if 1 <= fi <= max_f]
        if args.only_existing:
            sel = sorted(set(selected) & set(existing_fidx))
        else:
            sel = sorted(set(selected))
    else:
        # 預設列出 1..max_f；--only-existing 則使用實際存在之 fidx 清單
        if args.only_existing:
            sel = existing_fidx
        else:
            sel = list(range(1, max_f + 1))

    md = build_table(args.dim, args.result_dir, sel)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(md)
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
