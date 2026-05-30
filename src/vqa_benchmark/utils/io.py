import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    path = Path(path)
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def write_jsonl(items: Iterable[Dict[str, Any]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
