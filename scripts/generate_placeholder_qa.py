import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/qa/placeholder_qa.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = [
        {
            "qa_id": "placeholder_000001",
            "image_id": "PLACEHOLDER_IMAGE_ID",
            "image_path": "test/PLACEHOLDER_IMAGE_ID.dicom",
            "question_type": "specific_finding_presence",
            "question": "Is there pleural effusion?",
            "answer": "yes",
            "finding": "Pleural effusion",
            "bbox": [0, 0, 0, 0],
            "split": "test",
            "metadata": {
                "source": "placeholder",
                "note": "Replace this with deterministic QA generated from VinDr-CXR bbox annotations."
            }
        },
        {
            "qa_id": "placeholder_000002",
            "image_id": "PLACEHOLDER_IMAGE_ID",
            "image_path": "test/PLACEHOLDER_IMAGE_ID.dicom",
            "question_type": "local_abnormality_presence",
            "question": "Is there any local abnormality in this chest X-ray?",
            "answer": "yes",
            "finding": "any_local_abnormality",
            "bbox": [0, 0, 0, 0],
            "split": "test",
            "metadata": {
                "source": "placeholder",
                "note": "Replace this with deterministic QA generated from VinDr-CXR bbox annotations."
            }
        }
    ]

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved placeholder QA to {output_path}")


if __name__ == "__main__":
    main()
