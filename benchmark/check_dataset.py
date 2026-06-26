#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import run as bench


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local Indory waybill benchmark labels.")
    parser.add_argument("--dataset", default=bench.DEFAULT_DATASET)
    parser.add_argument("--dataset-root", type=Path, default=bench.DEFAULT_DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="fail if any selected image lacks expected labels")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args(argv)


def check_dataset(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    images, expected, info = bench.resolve_dataset(
        args.dataset,
        args.dataset_root.expanduser(),
        limit=args.limit,
    )
    missing_images = [str(path) for path in images if not path.exists()]
    unlabeled_images = [str(path) for path in images if bench.expected_for_image(expected, path) is None]
    expected_records = {
        id(record)
        for record in expected.values()
        if isinstance(record, dict)
    }
    summary = {
        "ok": not missing_images and (not args.strict or not unlabeled_images),
        "dataset": info,
        "image_count": len(images),
        "existing_image_count": len(images) - len(missing_images),
        "missing_image_count": len(missing_images),
        "expected_record_count": len(expected_records),
        "selected_unlabeled_count": len(unlabeled_images),
        "strict": bool(args.strict),
        "missing_images": missing_images[:20],
        "selected_unlabeled_images": unlabeled_images[:20],
    }
    return (0 if summary["ok"] else 1), summary


def print_human(summary: dict[str, Any]) -> None:
    dataset = summary.get("dataset") or {}
    print(f"dataset: {dataset.get('name')}")
    print(f"source: {dataset.get('source')}")
    print(f"expected: {dataset.get('expected_source') or '-'}")
    print(f"images: {summary.get('existing_image_count')}/{summary.get('image_count')}")
    print(f"expected records: {summary.get('expected_record_count')}")
    print(f"selected unlabeled: {summary.get('selected_unlabeled_count')}")
    if summary.get("missing_image_count"):
        print("missing images:")
        for path in summary.get("missing_images") or []:
            print(f"  {path}")
    if summary.get("selected_unlabeled_count"):
        print("selected images without labels:")
        for path in summary.get("selected_unlabeled_images") or []:
            print(f"  {path}")
    print("ok" if summary.get("ok") else "failed")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        code, summary = check_dataset(args)
    except Exception as exc:
        print(f"dataset check failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json_dumps(summary))
    else:
        print_human(summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
