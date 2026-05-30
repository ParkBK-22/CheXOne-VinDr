import argparse
import json
from pathlib import Path

from vqa_benchmark.evaluation.metrics import (
    compute_accuracy,
    compute_flip_rate,
    summarize_by_question_type,
)
from vqa_benchmark.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--lesion", type=str, default=None)
    parser.add_argument("--non-lesion", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    original_rows = read_jsonl(args.original)

    metrics = {
        "num_original": len(original_rows),
        "original_accuracy": compute_accuracy(original_rows),
        "by_question_type": summarize_by_question_type(original_rows),
    }

    if args.lesion:
        lesion_rows = read_jsonl(args.lesion)
        metrics["num_lesion"] = len(lesion_rows)
        metrics["lesion_accuracy"] = compute_accuracy(lesion_rows)
        metrics["lesion_flip_rate"] = compute_flip_rate(original_rows, lesion_rows)

    if args.non_lesion:
        non_lesion_rows = read_jsonl(args.non_lesion)
        metrics["num_non_lesion"] = len(non_lesion_rows)
        metrics["non_lesion_accuracy"] = compute_accuracy(non_lesion_rows)
        metrics["non_lesion_flip_rate"] = compute_flip_rate(original_rows, non_lesion_rows)

    if "lesion_flip_rate" in metrics and "non_lesion_flip_rate" in metrics:
        metrics["faithfulness_gap"] = (
            metrics["lesion_flip_rate"] - metrics["non_lesion_flip_rate"]
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
