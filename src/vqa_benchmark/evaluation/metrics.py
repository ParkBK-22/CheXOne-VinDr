from collections import defaultdict
from typing import Dict, List

from vqa_benchmark.evaluation.parser import normalize_answer, parse_yes_no


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


def yes_no_match(pred: str, gt: str) -> bool:
    return parse_yes_no(pred) == parse_yes_no(gt)


def compute_accuracy(
    rows: List[Dict],
    answer_key: str = "answer",
    pred_key: str = "prediction",
) -> float:
    if not rows:
        return 0.0

    correct = 0

    for row in rows:
        qtype = row.get("question_type", "")
        gt = row.get(answer_key, "")
        pred = row.get(pred_key, "")

        if "presence" in qtype or normalize_answer(gt) in {"yes", "no"}:
            correct += yes_no_match(pred, gt)
        else:
            correct += exact_match(pred, gt)

    return correct / len(rows)


def compute_flip_rate(
    original_rows: List[Dict],
    perturbed_rows: List[Dict],
    pred_key: str = "prediction",
) -> float:
    original_by_id = {r["qa_id"]: r for r in original_rows}
    perturbed_by_id = {r["qa_id"]: r for r in perturbed_rows}

    common_ids = sorted(set(original_by_id) & set(perturbed_by_id))

    if not common_ids:
        return 0.0

    flips = 0

    for qid in common_ids:
        original_pred = normalize_answer(original_by_id[qid].get(pred_key, ""))
        perturbed_pred = normalize_answer(perturbed_by_id[qid].get(pred_key, ""))
        flips += original_pred != perturbed_pred

    return flips / len(common_ids)


def summarize_by_question_type(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    groups = defaultdict(list)

    for row in rows:
        groups[row.get("question_type", "unknown")].append(row)

    return {
        qtype: {
            "num_samples": len(items),
            "accuracy": compute_accuracy(items),
        }
        for qtype, items in groups.items()
    }
