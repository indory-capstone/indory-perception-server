#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = Path.home() / "data" / "benchmarks" / "waybill_ocr" / "run_full_640x480" / "current_manifest.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "benchmark" / "datasets"
DEFAULT_EXPORT_DIR = DEFAULT_DATASET_ROOT / "indory_waybill_ocr_640x480"
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


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"manifest must be a list or contain samples: {path}")
    meta = payload if isinstance(payload, dict) else {}
    return meta, [sample for sample in samples if isinstance(sample, dict)]


def clean_ground_truth(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    cleaned = {
        key: value.get(key)
        for key in ("destination_room", "destination_floor", "destination_dong")
        if value.get(key) not in (None, "")
    }
    return cleaned or None


def sample_image_path(sample: dict[str, Any]) -> Path | None:
    for key in ("image_path", "image", "path", "file_name", "file"):
        value = sample.get(key)
        if value:
            return Path(str(value)).expanduser()
    return None


def safe_slug(value: Any, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    allowed = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    slug = "".join(allowed).strip("._")
    return slug or fallback


def unique_image_name(sample: dict[str, Any], image_path: Path, used: set[str]) -> str:
    name = safe_slug(image_path.name, "image")
    if not image_path.suffix.lower() in IMAGE_SUFFIXES:
        name = f"{safe_slug(image_path.stem, 'image')}.jpg"
    if name not in used:
        used.add(name)
        return name
    prefix = safe_slug(sample.get("sample_id"), image_path.stem)
    candidate = f"{prefix}_{name}"
    counter = 2
    while candidate in used:
        candidate = f"{prefix}_{counter}_{name}"
        counter += 1
    used.add(candidate)
    return candidate


def prune_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path, paths: list[Path]) -> None:
    rows = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def dataset_card(name: str, summary: dict[str, Any]) -> str:
    return f"""---
task_categories:
- image-to-text
language:
- ko
tags:
- ocr
- waybill
- robotics
- indory
- synthetic
license: other
---

# {name}

Indory waybill OCR benchmark dataset.

This dataset contains resized waybill images and destination labels used to
evaluate the `control-server-detection` `/v1/waybill/scan` API. The service's
internal Python package is still named `indory_ocr` for compatibility, but the
public repository name is `control-server-detection`. The benchmark target is
the delivery destination room/floor decision, not generic full-document OCR.

## Data Provenance and Privacy

All waybill images in this export are generated/synthetic benchmark images.
They do not contain real customer names, real phone numbers, real addresses,
real tracking numbers, service credentials, private network details, or robot
runtime logs. Any waybill-style identifiers visible in the images are synthetic
test text for OCR evaluation.

## Files

- `images/test/`: benchmark images
- `metadata.jsonl`: Hugging Face imagefolder metadata
- `manifest.json`: manifest consumed by `python3 benchmark/run.py`
- `dataset_summary.json`: export counts and source manifest metadata
- `checksums.sha256`: file integrity checksums

## Current Export

- Samples: {summary.get("sample_count")}
- Ground-truth samples: {summary.get("ground_truth_count")}
- Split: `{summary.get("split")}`
- Image size: 640x480 resized synthetic benchmark snapshot

Review the dataset license before reusing or redistributing the images.
"""


def export_dataset(
    *,
    source_manifest: Path,
    out_dir: Path,
    name: str,
    split: str,
    overwrite: bool,
    include_unscored: bool,
) -> dict[str, Any]:
    source_manifest = source_manifest.expanduser()
    out_dir = out_dir.expanduser()
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output dir already exists: {out_dir}")
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)

    source_meta, source_samples = load_manifest(source_manifest)
    samples: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    copied_files: list[Path] = []
    used_names: set[str] = set()
    skipped: list[dict[str, Any]] = []

    for sample in source_samples:
        ground_truth = clean_ground_truth(sample.get("ground_truth"))
        if not include_unscored and ground_truth is None:
            skipped.append({"sample_id": sample.get("sample_id"), "reason": "missing_ground_truth"})
            continue
        src_image = sample_image_path(sample)
        if src_image is None or not src_image.exists():
            skipped.append({"sample_id": sample.get("sample_id"), "reason": "missing_image", "image_path": str(src_image)})
            continue
        if src_image.suffix.lower() not in IMAGE_SUFFIXES:
            skipped.append({"sample_id": sample.get("sample_id"), "reason": "unsupported_image_type", "image_path": str(src_image)})
            continue

        image_name = unique_image_name(sample, src_image, used_names)
        rel_image = Path("images") / split / image_name
        dst_image = out_dir / rel_image
        shutil.copy2(src_image, dst_image)
        copied_files.append(dst_image)

        record = prune_none(
            {
                "sample_id": sample.get("sample_id") or Path(image_name).stem,
                "file_name": rel_image.as_posix(),
                "image_path": rel_image.as_posix(),
                "split": split,
                "ground_truth": ground_truth,
                "has_ground_truth": bool(ground_truth),
                "benchmark_status": sample.get("benchmark_status"),
                "benchmark_annotation_source": sample.get("benchmark_annotation_source"),
                "carrier": sample.get("carrier"),
                "condition": sample.get("condition"),
                "input_size": sample.get("input_size"),
                "original_size": sample.get("original_size"),
                "resize_mode": sample.get("resize_mode"),
                "source_container": sample.get("source_container"),
                "seed_id": sample.get("seed_id"),
                "qc_status": sample.get("qc_status"),
                "qc_grade": sample.get("qc_grade"),
            }
        )
        samples.append(record)
        metadata_rows.append(
            prune_none(
                {
                    "file_name": rel_image.as_posix(),
                    "sample_id": record.get("sample_id"),
                    "destination_room": (ground_truth or {}).get("destination_room"),
                    "destination_floor": (ground_truth or {}).get("destination_floor"),
                    "destination_dong": (ground_truth or {}).get("destination_dong"),
                    "benchmark_status": record.get("benchmark_status"),
                    "carrier": record.get("carrier"),
                    "condition": record.get("condition"),
                }
            )
        )

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "name": name,
        "dataset_format": "indory_waybill_ocr_hf_v1",
        "generated_at": generated_at,
        "source_manifest": source_manifest.name,
        "split": split,
        "samples": samples,
    }
    summary = {
        "name": name,
        "dataset_format": "indory_waybill_ocr_hf_v1",
        "generated_at": generated_at,
        "source_manifest": source_manifest.name,
        "source_manifest_sample_count": len(source_samples),
        "sample_count": len(samples),
        "ground_truth_count": sum(1 for sample in samples if sample.get("has_ground_truth")),
        "skipped_count": len(skipped),
        "split": split,
        "source_manifest_meta": {
            key: source_meta.get(key)
            for key in (
                "name",
                "generated_at",
                "current_image_count",
                "excluded_count",
                "missing_from_image_dir_count",
                "source_manifest_sample_count",
            )
            if key in source_meta
        },
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "dataset_summary.json", summary)
    write_jsonl(out_dir / "metadata.jsonl", metadata_rows)
    write_jsonl(out_dir / "skipped.jsonl", skipped)
    (out_dir / "README.md").write_text(dataset_card(name, summary), encoding="utf-8")
    write_checksums(
        out_dir,
        copied_files
        + [
            out_dir / "manifest.json",
            out_dir / "dataset_summary.json",
            out_dir / "metadata.jsonl",
            out_dir / "README.md",
        ],
    )
    return summary


def validate_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir.expanduser() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest.json: {manifest_path}")
    _meta, samples = load_manifest(manifest_path)
    missing_images: list[str] = []
    missing_gt: list[str] = []
    for sample in samples:
        image = sample_image_path(sample)
        if image is None:
            missing_images.append(str(sample.get("sample_id") or "<unknown>"))
            continue
        image_path = image if image.is_absolute() else manifest_path.parent / image
        if not image_path.exists():
            missing_images.append(str(image_path))
        if clean_ground_truth(sample.get("ground_truth")) is None:
            missing_gt.append(str(sample.get("sample_id") or image_path.name))
    return {
        "ok": not missing_images and not missing_gt,
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "missing_image_count": len(missing_images),
        "missing_ground_truth_count": len(missing_gt),
        "missing_images": missing_images[:20],
        "missing_ground_truth": missing_gt[:20],
    }


def download_dataset(*, repo_id: str, out_dir: Path, revision: str | None, force: bool) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("huggingface_hub is required; pip install huggingface-hub") from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=out_dir.expanduser(),
            force_download=force,
            allow_patterns=[
                "README.md",
                "manifest.json",
                "metadata.jsonl",
                "dataset_summary.json",
                "checksums.sha256",
                "images/**",
            ],
        )
    )
    validation = validate_dataset_dir(snapshot_path)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": str(snapshot_path),
        "manifest": str(snapshot_path / "manifest.json"),
        "validation": validation,
    }


def upload_dataset(
    *,
    dataset_dir: Path,
    repo_id: str,
    private: bool,
    create_repo: bool,
    commit_message: str,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("huggingface_hub is required; pip install huggingface-hub") from exc

    dataset_dir = dataset_dir.expanduser()
    validation = validate_dataset_dir(dataset_dir)
    if not validation["ok"]:
        raise ValueError(f"dataset validation failed: {json_dumps(validation)}")

    api = HfApi()
    repo_url = None
    if create_repo:
        repo_url = api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        commit_message=commit_message,
        ignore_patterns=[".DS_Store", "__pycache__/**"],
    )
    return {
        "repo_id": repo_id,
        "repo_url": str(repo_url) if repo_url is not None else None,
        "commit_url": getattr(commit, "commit_url", None),
        "commit_oid": getattr(commit, "oid", None),
        "validation": validation,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package, upload, and download Indory waybill OCR benchmark datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="copy images and labels into an HF-friendly dataset folder")
    export.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    export.add_argument("--out-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    export.add_argument("--name", default="indory_waybill_ocr_640x480")
    export.add_argument("--split", default="test")
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--include-unscored", action="store_true", help="include samples without ground_truth")

    validate = subparsers.add_parser("validate", help="validate a packaged dataset folder")
    validate.add_argument("--dataset-dir", type=Path, default=DEFAULT_EXPORT_DIR)

    upload = subparsers.add_parser("upload", help="upload a packaged dataset folder to Hugging Face")
    upload.add_argument("--dataset-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    upload.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. Fnhid/indory-waybill-ocr-640x480")
    upload.add_argument("--private", action="store_true")
    upload.add_argument("--create-repo", action="store_true")
    upload.add_argument("--commit-message", default="Upload Indory waybill OCR benchmark dataset")

    download = subparsers.add_parser("download", help="download an HF dataset snapshot for local eval")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--out-dir", type=Path, default=DEFAULT_DATASET_ROOT / "hf_download")
    download.add_argument("--revision")
    download.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "export":
            result = export_dataset(
                source_manifest=args.source_manifest,
                out_dir=args.out_dir,
                name=args.name,
                split=args.split,
                overwrite=args.overwrite,
                include_unscored=args.include_unscored,
            )
            result["out_dir"] = str(args.out_dir.expanduser())
            result["eval_command"] = f"python3 benchmark/run.py --dataset {args.out_dir.expanduser() / 'manifest.json'}"
        elif args.command == "validate":
            result = validate_dataset_dir(args.dataset_dir)
        elif args.command == "upload":
            result = upload_dataset(
                dataset_dir=args.dataset_dir,
                repo_id=args.repo_id,
                private=args.private,
                create_repo=args.create_repo,
                commit_message=args.commit_message,
            )
        elif args.command == "download":
            result = download_dataset(
                repo_id=args.repo_id,
                out_dir=args.out_dir,
                revision=args.revision,
                force=args.force,
            )
            result["eval_command"] = f"python3 benchmark/run.py --dataset {Path(result['manifest'])}"
            result["direct_hf_eval_command"] = f"python3 benchmark/run.py --dataset hf:{args.repo_id}"
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"dataset command failed: {exc}", file=sys.stderr)
        return 2

    print(json_dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
