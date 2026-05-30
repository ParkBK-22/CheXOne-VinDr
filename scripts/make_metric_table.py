import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


def normalize(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s/_.-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def map_option_letter_to_text(pred, options):
    if not options:
        return str(pred)

    pred_norm = normalize(pred)
    if not pred_norm:
        return str(pred)

    first = pred_norm.split()[0].strip(".):-(")
    if len(first) == 1 and first.isalpha():
        idx = ord(first.upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])

    return str(pred)


def parse_pred(row):
    pred = row.get("prediction", "")

    # Set1 multiple-choice case
    options = row.get("options_used") or row.get("options")
    if options:
        return map_option_letter_to_text(pred, options)

    # Set2 open-ended case
    return str(pred)


def is_correct(row):
    gt = normalize(row.get("answer", ""))
    pred = normalize(row.get("parsed_prediction", ""))

    qtype = row.get("qtype") or row.get("question_type")

    # Yes/No
    if gt in {"yes", "no"}:
        return pred.startswith(gt)

    # Count
    if qtype == "Q3_count":
        pred_tokens = pred.replace(".", " ").split()
        return gt in pred_tokens

    # Open-ended / MC text
    return gt == pred or (gt and gt in pred)


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["qtype"] = r.get("qtype") or r.get("question_type")
            r["parsed_prediction"] = parse_pred(r)
            r["is_correct"] = is_correct(r)
            rows.append(r)
    return rows


def summarize(rows, set_name):
    out = []

    # Overall
    n = len(rows)
    acc = sum(r["is_correct"] for r in rows) / n if n else 0
    out.append({
        "set": set_name,
        "qtype": "Overall",
        "num_samples": n,
        "accuracy": acc,
    })

    # By qtype
    groups = defaultdict(list)
    for r in rows:
        groups[r["qtype"]].append(r)

    for qtype in sorted(groups.keys()):
        items = groups[qtype]
        n = len(items)
        acc = sum(r["is_correct"] for r in items) / n if n else 0
        out.append({
            "set": set_name,
            "qtype": qtype,
            "num_samples": n,
            "accuracy": acc,
        })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set1", type=str, required=True)
    parser.add_argument("--set2", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/tables/metric_summary.csv")
    args = parser.parse_args()

    set1_rows = load_rows(args.set1)
    set2_rows = load_rows(args.set2)

    rows = []
    rows.extend(summarize(set1_rows, "Set1_options"))
    rows.extend(summarize(set2_rows, "Set2_open_rough"))

    df = pd.DataFrame(rows)
    df["accuracy_percent"] = (df["accuracy"] * 100).round(2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df[["set", "qtype", "num_samples", "accuracy_percent"]].to_string(index=False))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
