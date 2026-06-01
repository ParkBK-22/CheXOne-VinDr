#!/usr/bin/env python3
# ============================================================================
#  compute_flip.py — perturbation flip-rate 집계 (조건 A / B)
# ----------------------------------------------------------------------------
#  입력 preds jsonl 레코드(표준 스키마):
#     {"qid","image_id","qtype","condition","gt","pred"}
#     condition ∈ {"original","A","B"}.  같은 qid 가 original + (A|B) 로 2줄 존재.
#  → 같은 질문을 원본 vs 조작 이미지에 추론한 결과를 qid 로 짝지어 flip 을 센다.
#
#  출력: (조건 × qtype) flip 표 + (선택) summary csv.
#    - orig_acc    : 원본 이미지에서의 정확도
#    - flip_rate   : 원본 답 ≠ 조작 답 비율 (답이 바뀐 비율)
#    - to_expected : (조건 A 한정) 병변 제거 후 '정답'으로 flip 한 비율
#                    Q1→No, Q3→0, Q5→No specific disease
#
#  핵심 해석:
#    A 의 Q1 flip(Yes→No) 이 높고 B 의 Q1 flip(No→Yes) 이 낮으면 → visual grounding.
#    Δ = A flip − B flip 이 grounding 의 순효과(artifact 보정).
# ============================================================================
import argparse, json, collections, csv

EXPECT_A = {"Q1_abnormality": "no", "Q3_count": "0", "Q5_disease": "no specific disease"}


def norm(s):
    return str(s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="{qid,image_id,qtype,condition,gt,pred} jsonl")
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args()

    by_qid = collections.defaultdict(dict)
    for l in open(a.preds):
        r = json.loads(l)
        by_qid[r["qid"]][r["condition"]] = r

    agg = collections.defaultdict(lambda: {"n": 0, "flip": 0, "to_expect": 0, "orig_correct": 0})
    for qid, d in by_qid.items():
        if "original" not in d:
            continue
        o = d["original"]
        for cond in ("A", "B"):
            if cond not in d:
                continue
            p = d[cond]; qt = o["qtype"]; g = agg[(cond, qt)]
            g["n"] += 1
            g["orig_correct"] += int(norm(o["pred"]) == norm(o["gt"]))
            g["flip"] += int(norm(p["pred"]) != norm(o["pred"]))
            if cond == "A" and qt in EXPECT_A:
                g["to_expect"] += int(norm(p["pred"]) == EXPECT_A[qt])

    order = ["Q1_abnormality", "Q3_count", "Q5_disease", "Q2_finding", "Q4_location"]
    print(f"{'cond':4} {'qtype':16} {'N':>5} {'orig_acc':>9} {'flip_rate':>10} {'to_expected':>12}")
    out = []
    for cond in ("A", "B"):
        for qt in order:
            if (cond, qt) not in agg:
                continue
            g = agg[(cond, qt)]; n = g["n"]
            oa, fr = g["orig_correct"] / n, g["flip"] / n
            te = (g["to_expect"] / n) if (cond == "A" and qt in EXPECT_A) else float("nan")
            print(f"{cond:4} {qt:16} {n:>5} {oa:>9.3f} {fr:>10.3f} {te:>12.3f}")
            out.append({"condition": cond, "qtype": qt, "N": n, "orig_acc": round(oa, 4),
                        "flip_rate": round(fr, 4), "to_expected": (round(te, 4) if te == te else "")})

    aq1, bq1 = agg.get(("A", "Q1_abnormality")), agg.get(("B", "Q1_abnormality"))
    print("\n=== 핵심 (Q1 abnormality) ===")
    if aq1:
        print(f"  A 병변제거 Yes→No flip (grounding): {aq1['flip']/aq1['n']:.1%}  (to 'No': {aq1['to_expect']/aq1['n']:.1%})")
    if bq1:
        print(f"  B 정상조작 flip (artifact FP, 낮아야): {bq1['flip']/bq1['n']:.1%}")
    if aq1 and bq1:
        print(f"  Δ(A − B) grounding 순효과: {(aq1['flip']/aq1['n']) - (bq1['flip']/bq1['n']):.1%}")

    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "qtype", "N", "orig_acc", "flip_rate", "to_expected"])
            w.writeheader(); w.writerows(out)
        print(f"\nsaved {a.out_csv}")


if __name__ == "__main__":
    main()
