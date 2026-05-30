import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-dir", type=str, default="outputs/metrics")
    parser.add_argument("--output", type=str, default="outputs/tables/summary.csv")
    args = parser.parse_args()

    rows = []

    for path in Path(args.metric_dir).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"file": path.name}

        for key, value in data.items():
            if isinstance(value, (int, float, str)):
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df)
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
