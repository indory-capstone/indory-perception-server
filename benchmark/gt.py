#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path.home() / "data" / "benchmarks" / "waybill_ocr"
DEFAULT_RUN_DIR = DEFAULT_DATASET_ROOT / "run_full_640x480"
DEFAULT_SOURCE_MANIFEST = DEFAULT_RUN_DIR / "benchmark_manifest.json"
DEFAULT_IMAGE_DIR = DEFAULT_RUN_DIR / "resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "benchmark" / "ground_truth" / "current_annotations.jsonl"
DEFAULT_OUTPUT_MANIFEST = DEFAULT_RUN_DIR / "current_manifest.json"

REVIEW_FILES = {
    "low_confidence": "review_low_confidence.jsonl",
    "unreadable": "review_unreadable.jsonl",
    "exclude": "review_excluded_non_waybill.jsonl",
    "no_destination_detail": "review_unscored.jsonl",
    "unannotated": "review_unannotated.jsonl",
}
SCORABLE_STATUSES = {"verified"}
UNSCORED_STATUSES = {"low_confidence", "unreadable", "no_destination_detail", "unannotated"}
EXCLUDE_STATUSES = {"exclude"}
ALL_STATUSES = SCORABLE_STATUSES | UNSCORED_STATUSES | EXCLUDE_STATUSES
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def load_manifest_samples(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(path)
    samples = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError(f"manifest must be a list or contain samples: {path}")
    meta = data if isinstance(data, dict) else {}
    return meta, [sample for sample in samples if isinstance(sample, dict)]


def current_image_names(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir does not exist: {image_dir}")
    return {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            sample_id = str(record.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"missing sample_id at {path}:{line_no}")
            status = str(record.get("status") or "").strip()
            if status not in ALL_STATUSES:
                raise ValueError(f"unsupported status {status!r} at {path}:{line_no}")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id {sample_id!r} at {path}:{line_no}")
            if status in SCORABLE_STATUSES:
                ground_truth = record.get("ground_truth")
                if not isinstance(ground_truth, dict) or not any(
                    ground_truth.get(field)
                    for field in ("destination_room", "destination_floor", "destination_dong")
                ):
                    raise ValueError(f"verified annotation needs non-empty ground_truth at {path}:{line_no}")
            records[sample_id] = record
    return records


def clean_ground_truth(ground_truth: dict[str, Any]) -> dict[str, Any]:
    allowed = ("destination_room", "destination_floor", "destination_dong")
    cleaned = {key: ground_truth.get(key) for key in allowed if ground_truth.get(key) not in (None, "")}
    return cleaned


def sample_image_name(sample: dict[str, Any]) -> str | None:
    image = sample.get("image_path") or sample.get("image") or sample.get("path")
    return Path(str(image)).name if image else None


def review_record(sample: dict[str, Any], annotation: dict[str, Any], status: str) -> dict[str, Any]:
    record = {
        "sample_id": sample.get("sample_id"),
        "status": status,
        "image_path": sample.get("image_path"),
        "original_image_path": sample.get("original_image_path"),
    }
    for key in ("review_reason", "exclude_reason"):
        if annotation.get(key):
            record[key] = annotation[key]
    if isinstance(annotation.get("candidate_ground_truth"), dict):
        record["candidate_ground_truth"] = clean_ground_truth(annotation["candidate_ground_truth"])
    return record


def build_current_manifest(
    *,
    source_manifest: Path,
    image_dir: Path,
    annotations_path: Path,
    output_manifest: Path,
    strict_annotations: bool,
) -> dict[str, Any]:
    source_meta, source_samples = load_manifest_samples(source_manifest)
    annotations = load_annotations(annotations_path)
    present = current_image_names(image_dir)
    samples_by_id = {str(sample.get("sample_id")): sample for sample in source_samples if sample.get("sample_id")}

    unknown_annotations = sorted(sample_id for sample_id in annotations if sample_id not in samples_by_id)
    if unknown_annotations and strict_annotations:
        raise ValueError(f"annotation sample_id not found in source manifest: {unknown_annotations[:20]}")

    kept_samples: list[dict[str, Any]] = []
    review_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in REVIEW_FILES}
    missing_from_image_dir: list[dict[str, Any]] = []
    source_ground_truth_count = 0
    annotation_ground_truth_count = 0

    for sample in source_samples:
        image_name = sample_image_name(sample)
        if image_name not in present:
            missing_from_image_dir.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "image_path": sample.get("image_path"),
                    "image_name": image_name,
                }
            )
            continue

        sample_id = str(sample.get("sample_id") or "")
        annotation = annotations.get(sample_id)
        status = str(annotation.get("status")) if annotation else "unannotated"
        if status in EXCLUDE_STATUSES:
            review_rows["exclude"].append(review_record(sample, annotation or {}, status))
            continue

        item = dict(sample)
        item["image_path"] = str((image_dir / image_name).resolve())
        item["benchmark_status"] = status
        item["benchmark_annotation_source"] = "manual_jsonl" if annotation else "none"

        if annotation and status in SCORABLE_STATUSES:
            item["ground_truth"] = clean_ground_truth(annotation["ground_truth"])
            item["has_ground_truth"] = True
            annotation_ground_truth_count += 1
        elif isinstance(item.get("ground_truth"), dict):
            item["ground_truth"] = clean_ground_truth(item["ground_truth"])
            item["has_ground_truth"] = bool(item["ground_truth"])
            item["benchmark_status"] = "verified"
            item["benchmark_annotation_source"] = "source_manifest"
            source_ground_truth_count += int(bool(item["ground_truth"]))
        else:
            item["ground_truth"] = None
            item["has_ground_truth"] = False
            review_rows[status].append(review_record(sample, annotation or {}, status))

        kept_samples.append(item)

    annotated_present = {
        sample_id
        for sample_id, annotation in annotations.items()
        if sample_id in samples_by_id
        and sample_image_name(samples_by_id[sample_id]) in present
        and annotation.get("status") not in EXCLUDE_STATUSES
    }
    present_without_annotation = [
        str(sample.get("sample_id"))
        for sample in source_samples
        if sample.get("sample_id")
        and sample_image_name(sample) in present
        and str(sample.get("sample_id")) not in annotations
        and not isinstance(sample.get("ground_truth"), dict)
    ]
    if present_without_annotation and strict_annotations:
        raise ValueError(f"present samples without GT or annotation: {present_without_annotation[:20]}")

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    output = {
        "name": "indory_waybill_current_640x480",
        "generated_at": generated_at,
        "source_manifest": str(source_manifest),
        "image_dir": str(image_dir),
        "annotation_file": str(annotations_path),
        "source_manifest_sample_count": len(source_samples),
        "current_image_count": len(present),
        "missing_from_image_dir_count": len(missing_from_image_dir),
        "excluded_count": len(review_rows["exclude"]),
        "samples": kept_samples,
    }
    write_json(output_manifest, output)

    for status, filename in REVIEW_FILES.items():
        write_jsonl(output_manifest.parent / filename, review_rows[status])
    write_jsonl(output_manifest.parent / "missing_from_resized_images.jsonl", missing_from_image_dir)

    summary = {
        "generated_at": generated_at,
        "source_manifest": str(source_manifest),
        "image_dir": str(image_dir),
        "output_manifest": str(output_manifest),
        "annotation_file": str(annotations_path),
        "source_manifest_sample_count": len(source_samples),
        "current_image_count": len(present),
        "manifest_sample_count": len(kept_samples),
        "ground_truth_count": source_ground_truth_count + annotation_ground_truth_count,
        "source_ground_truth_count": source_ground_truth_count,
        "annotation_ground_truth_count": annotation_ground_truth_count,
        "excluded_count": len(review_rows["exclude"]),
        "missing_from_image_dir_count": len(missing_from_image_dir),
        "review_counts": {status: len(rows) for status, rows in review_rows.items()},
        "unknown_annotation_count": len(unknown_annotations),
        "present_without_annotation_count": len(present_without_annotation),
        "annotated_present_count": len(annotated_present),
        "review_files": {
            status: str(output_manifest.parent / filename)
            for status, filename in REVIEW_FILES.items()
        },
    }
    write_json(output_manifest.parent / "current_gt_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current Indory waybill benchmark GT files.")
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="create current manifest and review JSONL files")
    build.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    build.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    build.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    build.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    build.add_argument(
        "--allow-unannotated",
        action="store_true",
        help="allow present samples without source GT or manual annotation",
    )
    parser.set_defaults(command="build")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "build":
        raise ValueError(f"unsupported command: {args.command}")
    summary = build_current_manifest(
        source_manifest=args.source_manifest.expanduser(),
        image_dir=args.image_dir.expanduser(),
        annotations_path=args.annotations.expanduser(),
        output_manifest=args.output_manifest.expanduser(),
        strict_annotations=not args.allow_unannotated,
    )
    print(json_dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
