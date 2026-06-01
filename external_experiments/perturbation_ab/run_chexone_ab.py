#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

from tqdm import tqdm

from vqa_benchmark.models import build_model


def normalize(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s/_.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_option_letter(pred, options):
    if not options:
        return str(pred)

    raw = str(pred).strip()
    pred_norm = normalize(raw)

    # already option text
    for opt in options:
        if normalize(opt) == pred_norm:
            return str(opt)

    # A / B / C ...
    m = re.search(r"\b([A-Ka-k])\b", raw)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])

    # A. Yes, B), (C), etc.
    m = re.match(r"^\s*[\(\[]?([A-Ka-k])[\)\]\.:]?", raw)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])

    return raw


def build_prompt(q):
    question = q["question"]
    options = q.get("options", None)

    if options:
        option_lines = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
        )
        return (
            f"{question}\n\n"
            f"Options:\n"
            f"{option_lines}\n\n"
            f"Answer with only the option text.\n\n"
            f"Answer briefly."
        )

    return f"{question}\n\nAnswer briefly."


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "qid",
        "image_id",
        "qtype",
        "condition",
        "gt",
        "raw_pred",
        "pred",
        "image_path",
        "error",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def run_condition(model, questions, perturb_dir, cond):
    assert cond in {"A", "B"}
    rows = []

    for q in tqdm(questions, desc=f"CheXOne {cond}"):
        image_id = q["image_id"]
        prompt = build_prompt(q)

        pairs = [
            ("original", Path(perturb_dir) / "original" / f"{image_id}.png"),
            (cond, Path(perturb_dir) / cond / f"{image_id}.png"),
        ]

        for condition, image_path in pairs:
            try:
                raw = model.answer(str(image_path), prompt)
                pred = map_option_letter(raw, q.get("options"))
                err = None
            except Exception as e:
                raw = ""
                pred = ""
                err = repr(e)

            rows.append({
                "qid": q["qid"],
                "image_id": image_id,
                "qtype": q["qtype"],
                "condition": condition,
                "gt": q["answer"],
                "raw_pred": raw,
                "pred": pred,
                "image_path": str(image_path),
                "error": err,
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb-dir", required=True)
    ap.add_argument("--model-id", default="StanfordAIMI/CheXOne")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--limit-a", type=int, default=None)
    ap.add_argument("--limit-b", type=int, default=None)
    args = ap.parse_args()

    qA_path = Path(args.perturb_dir) / "questions_A.jsonl"
    qB_path = Path(args.perturb_dir) / "questions_B.jsonl"

    qA = read_jsonl(qA_path)
    qB = read_jsonl(qB_path)

    if args.limit_a is not None:
        qA = qA[: args.limit_a]
    if args.limit_b is not None:
        qB = qB[: args.limit_b]

    model = build_model(
        "chexone",
        model_id=args.model_id,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=32,
    )

    rows = []
    rows.extend(run_condition(model, qA, args.perturb_dir, "A"))
    rows.extend(run_condition(model, qB, args.perturb_dir, "B"))

    write_jsonl(rows, args.out_jsonl)
    write_csv(rows, args.out_csv)

    n_err = sum(1 for r in rows if r.get("error"))
    print(f"saved jsonl: {args.out_jsonl}")
    print(f"saved csv:   {args.out_csv}")
    print(f"rows: {len(rows)}")
    print(f"errors: {n_err}")


if __name__ == "__main__":
    main()
