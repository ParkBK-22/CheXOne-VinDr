from collections import defaultdict
from typing import Dict, List, Optional

from vqa_benchmark.evaluation.parser import normalize_answer, parse_yes_no


def get_qtype(row: Dict) -> str:
    return row.get("question_type") or row.get("qtype") or "unknown"


def get_id(row: Dict) -> str:
    return str(row.get("qa_id") or row.get("qid"))


def map_option_letter_to_text(pred: str, options: Optional[List[str]]) -> Optional[str]:
    """
    Map model outputs like 'A', 'A.', 'A. Yes', '(A)' to the corresponding option text.
    This is needed because many VLMs answer MC questions with option letters.
    """
    if not options:
        return None

    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return None

    first = pred_norm.split()[0].strip(".):-(")

    if len(first) == 1 and first.isalpha():
        idx = ord(first.upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])

    return None


def canonical_prediction(row: Dict, pred_key: str = "prediction") -> str:
    pred = row.get(pred_key, "")

    mapped = map_option_letter_to_text(
        pred=pred,
        options=row.get("options_used") or row.get("options"),
    )

    if mapped is not None:
        return mapped

    return pred


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


def option_match(pred: str, gt: str) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)

    if pred_norm == gt_norm:
        return True

    # Allows outputs like "The answer is Cardiomegaly."
    if gt_norm and gt_norm in pred_norm:
        return True

    return False


def numeric_match(pred: str, gt: str) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)

    pred_tokens = pred_norm.replace(".", " ").split()
    gt_tokens = gt_norm.replace(".", " ").split()

    return bool(gt_tokens) and gt_tokens[0] in pred_tokens


def yes_no_match(pred: str, gt: str) -> bool:
    return parse_yes_no(pred) == parse_yes_no(gt)


def is_yes_no_gt(gt: str) -> bool:
    return normalize_answer(gt) in {"yes", "no"}


def compute_accuracy(
    rows: List[Dict],
    answer_key: str = "answer",
    pred_key: str = "prediction",
) -> float:
    if not rows:
        return 0.0

    correct = 0

    for row in rows:
        qtype = get_qtype(row)
        gt = row.get(answer_key, "")
        pred = canonical_prediction(row, pred_key=pred_key)

        if is_yes_no_gt(gt):
            correct += yes_no_match(pred, gt)
        elif qtype == "Q3_count":
            correct += numeric_match(pred, gt)
        else:
            correct += option_match(pred, gt)

    return correct / len(rows)


def compute_flip_rate(
    original_rows: List[Dict],
    perturbed_rows: List[Dict],
    pred_key: str = "prediction",
) -> float:
    original_by_id = {get_id(r): r for r in original_rows}
    perturbed_by_id = {get_id(r): r for r in perturbed_rows}

    common_ids = sorted(set(original_by_id) & set(perturbed_by_id))

    if not common_ids:
        return 0.0

    flips = 0

    for qid in common_ids:
        original_pred = normalize_answer(canonical_prediction(original_by_id[qid], pred_key))
        perturbed_pred = normalize_answer(canonical_prediction(perturbed_by_id[qid], pred_key))
        flips += original_pred != perturbed_pred

    return flips / len(common_ids)


def summarize_by_question_type(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    groups = defaultdict(list)

    for row in rows:
        groups[get_qtype(row)].append(row)

    return {
        qtype: {
            "num_samples": len(items),
            "accuracy": compute_accuracy(items),
        }
        for qtype, items in groups.items()
    }
