#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Markdown table from experiment_summary.csv")
    p.add_argument("--summary-csv", type=Path, default=Path("outputs/experiment_summary.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/experiment_summary_table.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    lines = [
        "| Layer | Mean Return (Last 50) | Std Across Seeds | Mean Return (All Episodes) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {layer} | {m:.4f} | {s:.4f} | {auc:.4f} |".format(
                layer=row["layer"],
                m=float(row["mean_last_50"]),
                s=float(row["std_last_50"]),
                auc=float(row["mean_auc"]),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
