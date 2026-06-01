"""Step 3 — 추론 결과 분석: qtype × condition 정확도 표 + summary CSV.

Usage:
    python 03_compare.py --results /workspace/pipeline_out/results_chexagent.jsonl
"""
import os, argparse, json, csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_*.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    if not rows:
        print("empty results"); return

    CONDS = list(rows[0]["preds"].keys())

    # accuracy table
    by_qt = defaultdict(list)
    for r in rows: by_qt[r["qtype"]].append(r)

    print("\n" + "="*72)
    print("Accuracy per qtype × condition")
    print("="*72)
    header = f"{'qtype':<18} {'N':>5}  " + "  ".join(f"{c:>14}" for c in CONDS)
    print(header)
    print("-"*len(header))
    summary = []
    for qt in ["Q1_abnormality","Q2_finding","Q3_count","Q4_location","Q5_disease"]:
        sub = by_qt.get(qt, [])
        if not sub: continue
        row_str = f"{qt:<18} {len(sub):>5}  "
        rec = {"qtype": qt, "N": len(sub)}
        for c in CONDS:
            ok = [r["correct"].get(c) for r in sub if r["correct"].get(c) is not None]
            n_eval = len(ok); n_correct = sum(ok)
            acc = 100*n_correct/n_eval if n_eval else 0
            row_str += f"  {n_correct:>4}/{n_eval:<4} ({acc:>4.1f}%)".rjust(14)
            rec[c] = round(acc, 1); rec[f"{c}_correct"] = n_correct; rec[f"{c}_n"] = n_eval
        print(row_str)
        summary.append(rec)
    # total
    print("-"*len(header))
    tot_str = f"{'TOTAL':<18} {len(rows):>5}  "
    for c in CONDS:
        ok = [r["correct"].get(c) for r in rows if r["correct"].get(c) is not None]
        n_eval = len(ok); n_correct = sum(ok)
        acc = 100*n_correct/n_eval if n_eval else 0
        tot_str += f"  {n_correct:>4}/{n_eval:<4} ({acc:>4.1f}%)".rjust(14)
    print(tot_str)

    # save summary CSV
    out_csv = args.results.replace(".jsonl", "_summary.csv")
    if summary:
        cols = ["qtype","N"] + [c for c in CONDS]
        with open(out_csv,"w",newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for s in summary:
                w.writerow([s[k] for k in cols])
        print(f"\nsummary -> {out_csv}")


if __name__ == "__main__":
    main()
