#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "CONTROL_SERVER_DETECTION_BENCHMARK_ROOT",
        os.environ.get("INDORY_OCR_BENCHMARK_ROOT", str(Path.home() / "data" / "benchmarks" / "waybill_ocr")),
    )
)
DEFAULT_HF_DATASET_ROOT = Path(
    os.environ.get(
        "CONTROL_SERVER_DETECTION_HF_DATASET_ROOT",
        os.environ.get("INDORY_OCR_HF_DATASET_ROOT", str(REPO_ROOT / "benchmark" / "datasets" / "hf")),
    )
)
DEFAULT_DATASET = "current"
MODE_ENDPOINTS = {
    "waybill": "/v1/waybill/scan",
}
MODE_ALIASES = {
    "waybill": "waybill",
    "ocr_llm": "waybill",
    "ocr+llm": "waybill",
    "llm": "waybill",
}
DATASET_PRESETS = {
    "current": {
        "kind": "manifest",
        "path": "run_full_640x480/current_manifest.json",
        "description": "Current resized_images snapshot; deleted files and non-waybill exclusions are reflected",
    },
    "current_verified": {
        "kind": "manifest",
        "path": "run_full_640x480/current_manifest.json",
        "description": "Current resized_images snapshot, limited to samples with verified ground_truth",
        "ground_truth_only": True,
    },
    "labeled82": {
        "kind": "manifest",
        "path": "run_full_640x480/benchmark_manifest.json",
        "description": "82 labeled-eval 640x480 waybill images from the full local dataset",
        "labeled_only": True,
    },
    "groundtruth83": {
        "kind": "manifest",
        "path": "run_full_640x480/benchmark_manifest.json",
        "description": "83 images with ground_truth, including one QC-excluded sample",
        "ground_truth_only": True,
    },
    "full235": {
        "kind": "manifest",
        "path": "run_full_640x480/benchmark_manifest.json",
        "description": "235 full 640x480 waybill images; only rows with ground_truth are evaluated",
    },
    "labeled45": {
        "kind": "manifest",
        "path": "rerun_labeled45_no_corrections_spacing_layout_v2_640x480/selected_manifest.json",
        "description": "45 labeled 640x480 box-scene images with ground_truth",
    },
    "full198": {
        "kind": "directory",
        "path": "run_box_only_640x480/resized_images",
        "expected": "run_box_only_640x480/evaluation_rows.csv",
        "description": "198 resized box-scene images; only labeled rows are evaluated",
    },
    "smoke5": {
        "kind": "manifest",
        "path": "smoke_5/benchmark_manifest.json",
        "description": "5-image legacy smoke manifest",
    },
}


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)


def now_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_modes(value: str) -> list[str]:
    modes: list[str] = []
    for part in str(value or "").split(","):
        key = part.strip().lower()
        if not key:
            continue
        mode = MODE_ALIASES.get(key)
        if mode is None:
            allowed = ", ".join(sorted(MODE_ENDPOINTS))
            raise argparse.ArgumentTypeError(f"unsupported mode {part!r}; use {allowed}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return modes


def parse_rotations(value: str | None) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    rotations: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            rotation = int(text) % 360
        except ValueError as exc:
            raise argparse.ArgumentTypeError("rotations must be comma-separated degrees") from exc
        if rotation not in {0, 90, 180, 270}:
            raise argparse.ArgumentTypeError("rotations must use only 0,90,180,270")
        if rotation not in rotations:
            rotations.append(rotation)
    return rotations or None


def parse_scalar(value: str) -> Any:
    text = str(value)
    lower = text.strip().lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def parse_option_assignments(values: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"--option must be KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("--option key cannot be empty")
        options[key] = parse_scalar(raw.strip())
    return options


def has_glob_magic(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def collect_images(inputs: list[str], *, recursive: bool, limit: int) -> list[Path]:
    images: list[Path] = []
    missing: list[str] = []

    for raw in inputs:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        candidates: list[Path] = []
        if has_glob_magic(expanded):
            candidates = [Path(path) for path in sorted(glob.glob(expanded, recursive=recursive))]
            if not candidates:
                missing.append(raw)
                continue
        else:
            candidates = [Path(expanded)]

        for path in candidates:
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.iterdir()
                images.extend(
                    sorted(
                        file_path
                        for file_path in iterator
                        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
                    )
                )
            elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
            elif not path.exists():
                missing.append(raw)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for image in images:
        try:
            key = image.resolve()
        except Exception:
            key = image
        if key in seen:
            continue
        seen.add(key)
        deduped.append(image)

    if missing:
        print(f"warning: skipped missing input(s): {', '.join(missing)}", file=sys.stderr)
    if limit > 0:
        deduped = deduped[:limit]
    return deduped


def image_key_variants(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    path = Path(text)
    stem = path.stem
    variants = [text, path.name, stem]
    if stem.endswith("_640x480"):
        variants.append(stem[: -len("_640x480")])
    if path.suffix:
        variants.append(path.with_suffix("").name)
    deduped: list[str] = []
    for variant in variants:
        if variant and variant not in deduped:
            deduped.append(variant)
    return deduped


def resolve_relative_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def resolve_record_image(record: dict[str, Any], base_dir: Path) -> Path | None:
    for field in ("image_path", "image", "path", "file_name", "file"):
        path = resolve_relative_path(record.get(field), base_dir)
        if path is not None:
            return path
    return None


def register_expected(records: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    keys: list[str] = []
    for field in ("image", "file", "path", "name", "sample_id"):
        keys.extend(image_key_variants(record.get(field)))
    for key in keys:
        records[key] = record


def image_format(path: Path) -> str:
    suffix = path.suffix.lower().strip(".")
    if suffix == "jpeg":
        return "jpg"
    return suffix or "jpg"


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any] | None, str]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None
            return int(response.status), parsed, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None
        return int(exc.code), parsed, raw
    except Exception as exc:
        return 0, None, str(exc)


def compact_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return "".join(text.split())


def loose_label(value: Any) -> str:
    text = compact_text(value).upper()
    return text.replace("호", "").replace("_", "-")


def normalize_floor(value: Any) -> str:
    text = loose_label(value)
    if text.isdigit():
        return f"{int(text)}F"
    if text.startswith("F") and text[1:].isdigit():
        return f"{int(text[1:])}F"
    return text


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def expected_from_ground_truth(sample: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    record = {
        "image": (
            sample.get("image_path")
            or sample.get("image")
            or sample.get("path")
            or sample.get("file_name")
            or sample.get("file")
        ),
        "sample_id": sample.get("sample_id") or ground_truth.get("sample_id"),
        "destination_room": ground_truth.get("destination_room"),
        "destination_floor": ground_truth.get("destination_floor"),
        "destination_dong": ground_truth.get("destination_dong"),
    }
    return {key: value for key, value in record.items() if value not in (None, "")}


def expected_from_manifest_item(item: dict[str, Any]) -> dict[str, Any] | None:
    ground_truth = item.get("ground_truth")
    if isinstance(ground_truth, dict):
        return expected_from_ground_truth(item, ground_truth)
    if any(item.get(field) for field in ("destination_room", "destination_floor", "destination_dong")):
        return {
            key: value
            for key, value in {
                "image": (
                    item.get("image")
                    or item.get("image_path")
                    or item.get("path")
                    or item.get("file_name")
                    or item.get("file")
                ),
                "sample_id": item.get("sample_id"),
                "destination_room": item.get("destination_room"),
                "destination_floor": item.get("destination_floor"),
                "destination_dong": item.get("destination_dong"),
            }.items()
            if value not in (None, "")
        }
    return None


def manifest_images(path: Path, *, labeled_only: bool, ground_truth_only: bool = False) -> list[Path]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    base_dir = path.expanduser().parent
    samples = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError(f"manifest must be a list or contain samples: {path}")
    images: list[Path] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        ground_truth = item.get("ground_truth")
        if labeled_only and item.get("is_labeled_eval") is not True:
            continue
        if ground_truth_only and not isinstance(ground_truth, dict):
            continue
        image = resolve_record_image(item, base_dir)
        if image:
            images.append(image)
    return images


def load_expected_json(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    base_dir = path.expanduser().parent
    samples = data.get("samples") if isinstance(data, dict) else data
    if isinstance(samples, dict):
        samples = [samples]
    if not isinstance(samples, list):
        raise ValueError(f"expected JSON must be a list or contain samples: {path}")

    records: dict[str, dict[str, Any]] = {}
    for item in samples:
        if not isinstance(item, dict):
            continue
        record = expected_from_manifest_item(item)
        if record is None:
            continue
        image = resolve_record_image(item, base_dir)
        if image is not None:
            record["image"] = str(image)
        register_expected(records, record)
    return records


def load_expected_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            if not any(record.get(field) for field in ("image", "file", "path", "name", "sample_id")):
                raise ValueError(f"expected image/file/path/name/sample_id key at {path}:{line_no}")
            register_expected(records, record)
    return records


def load_expected_csv(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            room = row.get("gt_room") or row.get("destination_room")
            floor = row.get("gt_floor") or row.get("destination_floor")
            dong = row.get("gt_dong") or row.get("destination_dong")
            if not any((room, floor, dong)):
                continue
            record = {
                "image": row.get("image") or row.get("image_path") or row.get("path"),
                "sample_id": row.get("sample_id"),
                "destination_room": room,
                "destination_floor": floor,
                "destination_dong": dong,
            }
            register_expected(records, {key: value for key, value in record.items() if value not in (None, "")})
    return records


def load_expected(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_expected_jsonl(path)
    if suffix == ".json":
        return load_expected_json(path)
    if suffix == ".csv":
        return load_expected_csv(path)
    raise ValueError(f"unsupported expected label file type: {path}")


def expected_for_image(expected: dict[str, dict[str, Any]], image: Path) -> dict[str, Any] | None:
    keys = image_key_variants(image)
    keys.extend(image_key_variants(image.resolve()) if image.exists() else [])
    for key in keys:
        if key in expected:
            return expected[key]
    return None


def dataset_preset_path(name: str, dataset_root: Path, key: str = "path") -> Path:
    preset = DATASET_PRESETS[name]
    return (dataset_root / str(preset[key])).expanduser()


def safe_hf_cache_name(repo_id: str, revision: str | None) -> str:
    text = repo_id.strip().replace("/", "__").replace(":", "_")
    if revision:
        text = f"{text}@{revision}".replace("/", "__").replace(":", "_")
    return text


def parse_hf_dataset_ref(dataset: str | None) -> tuple[str, str | None] | None:
    if not dataset:
        return None
    text = str(dataset).strip()
    if text.startswith("hf://"):
        ref = text[len("hf://") :]
    elif text.startswith("hf:"):
        ref = text[len("hf:") :]
    else:
        return None
    ref = ref.strip()
    if not ref:
        raise ValueError("HF dataset ref is empty; use hf:owner/name or hf://owner/name")
    revision = None
    if "@" in ref:
        ref, revision = ref.rsplit("@", 1)
        ref = ref.strip()
        revision = revision.strip() or None
    return ref, revision


def snapshot_hf_dataset(repo_id: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "huggingface_hub is required for hf: datasets; install control-server-detection[benchmark] "
            "or pip install huggingface-hub"
        ) from exc

    local_dir = DEFAULT_HF_DATASET_ROOT / safe_hf_cache_name(repo_id, revision)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=[
            "README.md",
            "manifest.json",
            "metadata.jsonl",
            "dataset_summary.json",
            "checksums.sha256",
            "images/**",
        ],
    )
    manifest = Path(snapshot_path) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"HF dataset snapshot does not contain manifest.json: {manifest}")
    return manifest


def resolve_dataset(
    dataset: str | None,
    dataset_root: Path,
    *,
    limit: int,
) -> tuple[list[Path], dict[str, dict[str, Any]], dict[str, Any] | None]:
    if not dataset:
        return [], {}, None
    hf_ref = parse_hf_dataset_ref(dataset)
    if hf_ref is not None:
        repo_id, revision = hf_ref
        source = snapshot_hf_dataset(repo_id, revision)
        images = manifest_images(source, labeled_only=False, ground_truth_only=True)
        expected = load_expected(source)
        if limit > 0:
            images = images[:limit]
        return images, expected, {
            "name": dataset,
            "description": "Hugging Face dataset snapshot",
            "source": str(source),
            "expected_source": str(source),
            "hf_repo_id": repo_id,
            "hf_revision": revision,
        }
    if dataset in DATASET_PRESETS:
        preset = DATASET_PRESETS[dataset]
        source = dataset_preset_path(dataset, dataset_root)
        expected_source = dataset_preset_path(dataset, dataset_root, "expected") if preset.get("expected") else source
        if preset["kind"] == "manifest":
            images = manifest_images(
                source,
                labeled_only=bool(preset.get("labeled_only")),
                ground_truth_only=bool(preset.get("ground_truth_only")),
            )
        else:
            images = collect_images([str(source)], recursive=False, limit=0)
        expected = load_expected(expected_source) if expected_source.exists() else {}
        if limit > 0:
            images = images[:limit]
        return images, expected, {
            "name": dataset,
            "description": preset["description"],
            "source": str(source),
            "expected_source": str(expected_source) if expected_source.exists() else None,
        }

    path = Path(dataset).expanduser()
    if path.is_dir():
        images = collect_images([str(path)], recursive=False, limit=limit)
        return images, {}, {"name": str(dataset), "source": str(path), "expected_source": None}
    if path.is_file() and path.suffix.lower() == ".json":
        images = manifest_images(path, labeled_only=False)
        expected = load_expected(path)
        if limit > 0:
            images = images[:limit]
        return images, expected, {"name": str(dataset), "source": str(path), "expected_source": str(path)}
    raise ValueError(f"unknown dataset preset or path: {dataset}")


def summarize_response(mode: str, response: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    timing = response.get("timing") if isinstance(response.get("timing"), dict) else {}
    decision = response.get("decision") if isinstance(response.get("decision"), dict) else {}
    return {
        "destination": response.get("destination"),
        "destination_dong": decision.get("destination_dong"),
        "destination_floor": decision.get("destination_floor"),
        "destination_room": decision.get("destination_room"),
        "confidence": decision.get("confidence"),
        "needs_manual_review": response.get("needs_manual_review"),
        "auto_accept": response.get("auto_accept"),
        "risk_reasons": response.get("risk_reasons") or decision.get("risk_reasons") or [],
        "ocr_seconds": timing.get("ocr_seconds"),
        "llm_seconds": timing.get("llm_seconds"),
        "total_seconds": timing.get("total_seconds"),
    }


def evaluate_waybill(summary: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    field_specs = [
        ("destination_room", loose_label, loose_label),
        ("destination_dong", loose_label, loose_label),
        ("destination_floor", normalize_floor, normalize_floor),
    ]
    for field, normalize_expected, normalize_actual in field_specs:
        if field not in expected:
            continue
        expected_value = normalize_expected(expected.get(field))
        actual_value = normalize_actual(summary.get(field))
        checks.append(
            {
                "field": field,
                "expected": expected.get(field),
                "actual": summary.get(field),
                "pass": bool(expected_value and expected_value == actual_value),
            }
        )
    for field in ("auto_accept", "needs_manual_review"):
        if field not in expected:
            continue
        checks.append(
            {
                "field": field,
                "expected": bool(expected.get(field)),
                "actual": summary.get(field),
                "pass": bool(summary.get(field)) == bool(expected.get(field)),
            }
        )
    destination_text = compact_text(summary.get("destination"))
    for expected_text in as_list(
        expected.get("destination_contains")
        or expected.get("waybill_must_contain")
        or expected.get("destination")
    ):
        needle = compact_text(expected_text)
        checks.append(
            {
                "field": "destination_contains",
                "expected": expected_text,
                "actual": summary.get("destination"),
                "pass": bool(needle and needle in destination_text),
            }
        )
    return evaluation_result(checks)


def evaluation_result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    if not checks:
        return {"evaluated": False, "pass": None, "checks": []}
    passed = all(bool(check.get("pass")) for check in checks)
    return {"evaluated": True, "pass": passed, "checks": checks}


def evaluate(mode: str, summary: dict[str, Any], expected: dict[str, Any] | None) -> dict[str, Any]:
    if not expected:
        return {"evaluated": False, "pass": None, "checks": []}
    return evaluate_waybill(summary, expected)


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def debug_ocr_texts(response: dict[str, Any] | None) -> list[str]:
    if not isinstance(response, dict):
        return []
    debug = response.get("debug")
    if not isinstance(debug, dict):
        return []
    ocr = debug.get("ocr")
    if not isinstance(ocr, dict):
        return []

    texts: list[str] = []

    def add_items(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item.get("text")))

    add_items(ocr.get("items"))
    for result in ocr.get("combined_results") or []:
        if isinstance(result, dict):
            add_items(result.get("ocr_items"))

    deduped: list[str] = []
    for text in texts:
        normalized = normalize_for_ocr_presence(text)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def normalize_for_ocr_presence(value: Any) -> str:
    text = compact_text(value).upper()
    text = text.replace("호", "")
    text = text.replace("_", "-")
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("L", "1")
    return re.sub(r"[^A-Z0-9가-힣-]", "", text)


def expected_present_in_ocr(expected_value: Any, texts: list[str]) -> tuple[bool | None, list[str]]:
    if expected_value in (None, ""):
        return None, []
    if not texts:
        return None, []
    needle = normalize_for_ocr_presence(expected_value)
    if not needle:
        return None, []
    needles = [needle]
    if needle.endswith("호"):
        needles.append(needle[:-1])
    needles = [item for item in dict.fromkeys(needles) if len(item) >= 3]
    hits = [
        text
        for text in texts
        if any(item in text for item in needles)
        or (len(text) >= max(3, len(needle) - 1) and text in needle)
    ]
    return bool(hits), hits[:5]


def analyze_failure(
    *,
    ok: bool,
    status_code: int,
    error: Any,
    summary: dict[str, Any],
    evaluation: dict[str, Any],
    expected: dict[str, Any] | None,
    response: dict[str, Any] | None,
    low_confidence_threshold: float,
) -> dict[str, Any] | None:
    failed_expected = evaluation.get("evaluated") and evaluation.get("pass") is False
    if ok and not failed_expected:
        return None

    reasons: list[str] = []
    details: dict[str, Any] = {
        "status_code": status_code,
        "confidence": summary.get("confidence"),
        "low_confidence_threshold": low_confidence_threshold,
        "needs_manual_review": summary.get("needs_manual_review"),
        "risk_reasons": summary.get("risk_reasons") or [],
    }

    if not ok:
        reasons.append("http_error")
        details["error"] = error

    confidence = float_or_none(summary.get("confidence"))
    if confidence is not None and confidence < low_confidence_threshold:
        reasons.append("low_confidence")
    if summary.get("needs_manual_review") is True:
        reasons.append("needs_manual_review")

    failed_checks = [check for check in evaluation.get("checks") or [] if check.get("pass") is False]
    if failed_checks:
        details["failed_checks"] = failed_checks
        failed_fields = {str(check.get("field")) for check in failed_checks}
        if "destination_room" in failed_fields and not summary.get("destination_room"):
            reasons.append("missing_destination_room")
        elif any(field in failed_fields for field in ("destination_room", "destination_dong", "destination_floor")):
            reasons.append("destination_mismatch")

    if expected:
        details["expected"] = {
            key: expected.get(key)
            for key in ("destination_dong", "destination_floor", "destination_room")
            if expected.get(key) not in (None, "")
        }
        details["actual"] = {
            "destination_dong": summary.get("destination_dong"),
            "destination_floor": summary.get("destination_floor"),
            "destination_room": summary.get("destination_room"),
        }

    texts = debug_ocr_texts(response)
    if expected and expected.get("destination_room"):
        present, hits = expected_present_in_ocr(expected.get("destination_room"), texts)
        details["ocr_contains_expected_room"] = present
        if hits:
            details["ocr_expected_room_hits"] = hits
        if present is False:
            reasons.append("ocr_miss_expected_room")
        elif present is True and failed_expected:
            reasons.append("selection_missed_visible_room")
        elif present is None and failed_expected:
            reasons.append("ocr_debug_unavailable")

    primary_order = [
        "http_error",
        "ocr_miss_expected_room",
        "low_confidence",
        "missing_destination_room",
        "selection_missed_visible_room",
        "destination_mismatch",
        "needs_manual_review",
        "ocr_debug_unavailable",
    ]
    deduped_reasons: list[str] = []
    for reason in reasons:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)
    primary = next((reason for reason in primary_order if reason in deduped_reasons), deduped_reasons[0] if deduped_reasons else "unknown")
    return {
        "primary_reason": primary,
        "reasons": deduped_reasons or ["unknown"],
        "details": details,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode")), []).append(row)

    modes: dict[str, dict[str, Any]] = {}
    for mode, mode_rows in sorted(by_mode.items()):
        latencies = [float(row.get("latency_s") or 0.0) for row in mode_rows if row.get("latency_s") is not None]
        evaluated = [row for row in mode_rows if (row.get("evaluation") or {}).get("evaluated")]
        expected_pass = [row for row in evaluated if (row.get("evaluation") or {}).get("pass") is True]
        ok_rows = [row for row in mode_rows if row.get("ok")]
        failure_rows = [
            row
            for row in mode_rows
            if (not row.get("ok")) or (row.get("evaluation") or {}).get("pass") is False
        ]
        failure_reason_counts: dict[str, int] = {}
        failure_samples: list[dict[str, Any]] = []
        for row in failure_rows:
            analysis = row.get("failure_analysis") if isinstance(row.get("failure_analysis"), dict) else {}
            reason = str(analysis.get("primary_reason") or "unknown")
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
            sample = {
                "image_name": row.get("image_name"),
                "primary_reason": reason,
            }
            details = analysis.get("details") if isinstance(analysis.get("details"), dict) else {}
            if details:
                sample["confidence"] = details.get("confidence")
                sample["needs_manual_review"] = details.get("needs_manual_review")
                sample["expected"] = details.get("expected")
                sample["actual"] = details.get("actual")
                sample["ocr_contains_expected_room"] = details.get("ocr_contains_expected_room")
            failure_samples.append(sample)
        mode_summary: dict[str, Any] = {
            "count": len(mode_rows),
            "ok_count": len(ok_rows),
            "error_count": len(mode_rows) - len(ok_rows),
            "expected_evaluated_count": len(evaluated),
            "expected_pass_count": len(expected_pass),
            "failure_count": len(failure_rows),
            "failure_reason_counts": failure_reason_counts,
            "failure_samples": failure_samples,
            "latency_s": {
                "mean": statistics.mean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "p90": percentile(latencies, 0.90),
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
        }
        if mode == "waybill":
            summaries = [row.get("summary") or {} for row in mode_rows]
            mode_summary.update(
                {
                    "auto_accept_count": sum(1 for item in summaries if item.get("auto_accept") is True),
                    "manual_review_count": sum(1 for item in summaries if item.get("needs_manual_review") is True),
                    "destinations": [item.get("destination") for item in summaries if item.get("destination")],
                }
            )
        modes[mode] = mode_summary

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_count": len(rows),
        "modes": modes,
    }


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def print_table(summary: dict[str, Any]) -> None:
    rows: list[list[str]] = []
    for mode, data in (summary.get("modes") or {}).items():
        latency = data.get("latency_s") or {}
        expected_eval = int(data.get("expected_evaluated_count") or 0)
        expected_pass = int(data.get("expected_pass_count") or 0)
        expected_cell = "-" if expected_eval <= 0 else f"{expected_pass}/{expected_eval}"
        rows.append(
            [
                mode,
                str(data.get("count") or 0),
                f"{data.get('ok_count') or 0}/{data.get('count') or 0}",
                expected_cell,
                format_float(latency.get("mean")),
                format_float(latency.get("median")),
                format_float(latency.get("p90")),
            ]
        )
    header = ["mode", "images", "http_ok", "expected", "avg_s", "p50_s", "p90_s"]
    widths = [len(item) for item in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    line = "  ".join(cell.ljust(width) for cell, width in zip(header, widths))
    sep = "  ".join("-" * width for width in widths)
    print(line)
    print(sep)
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def write_summary_md(path: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    expected_labels = manifest.get("expected_jsonl") or dataset.get("expected_source") or "-"
    lines = [
        "# Indory Waybill Benchmark",
        "",
        f"- Service URL: `{manifest.get('service_url')}`",
        f"- Dataset: `{dataset.get('name') or '-'}`",
        f"- Dataset source: `{dataset.get('source') or '-'}`",
        f"- Images: `{manifest.get('image_count')}`",
        f"- Modes: `{', '.join(manifest.get('modes') or [])}`",
        f"- Expected labels: `{expected_labels}`",
        "",
        "| Mode | Images | HTTP OK | Expected | Avg s | P50 s | P90 s | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode, data in (summary.get("modes") or {}).items():
        latency = data.get("latency_s") or {}
        expected_eval = int(data.get("expected_evaluated_count") or 0)
        expected_pass = int(data.get("expected_pass_count") or 0)
        expected_cell = "-" if expected_eval <= 0 else f"{expected_pass}/{expected_eval}"
        notes = ""
        if mode == "waybill":
            notes = (
                f"auto_accept={data.get('auto_accept_count') or 0}, "
                f"manual_review={data.get('manual_review_count') or 0}"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    str(data.get("count") or 0),
                    f"{data.get('ok_count') or 0}/{data.get('count') or 0}",
                    expected_cell,
                    format_float(latency.get("mean")),
                    format_float(latency.get("median")),
                    format_float(latency.get("p90")),
                    notes,
                ]
            )
            + " |"
        )
    failure_lines: list[str] = []
    for mode, data in (summary.get("modes") or {}).items():
        reason_counts = data.get("failure_reason_counts") or {}
        if not reason_counts:
            continue
        failure_lines.extend(
            [
                "",
                f"## Failure Reasons: {mode}",
                "",
                "| Reason | Count |",
                "| --- | ---: |",
            ]
        )
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            failure_lines.append(f"| {reason} | {count} |")

        samples = data.get("failure_samples") or []
        if samples:
            failure_lines.extend(
                [
                    "",
                    "| Image | Reason | Confidence | Expected | Actual | OCR contains expected room |",
                    "| --- | --- | ---: | --- | --- | --- |",
                ]
            )
            for sample in samples[:30]:
                expected = sample.get("expected") or {}
                actual = sample.get("actual") or {}
                expected_text = ", ".join(f"{key}={value}" for key, value in expected.items() if value not in (None, ""))
                actual_text = ", ".join(f"{key}={value}" for key, value in actual.items() if value not in (None, ""))
                failure_lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(sample.get("image_name") or ""),
                            str(sample.get("primary_reason") or "unknown"),
                            format_float(sample.get("confidence")),
                            expected_text or "-",
                            actual_text or "-",
                            str(sample.get("ocr_contains_expected_room")),
                        ]
                    )
                    + " |"
                )
    lines.extend(failure_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_if_present(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


def build_mode_options(args: argparse.Namespace) -> dict[str, Any]:
    options = parse_option_assignments(args.option or [])
    add_if_present(options, "judge_mode", args.judge_mode)
    add_if_present(options, "model_path", str(args.model_path) if args.model_path else None)
    add_if_present(options, "model", args.model)
    add_if_present(options, "endpoint", args.endpoint)
    add_if_present(options, "max_new_tokens", args.max_new_tokens)
    add_if_present(options, "llm_gpu_layers", args.llm_gpu_layers)
    add_if_present(options, "llm_threads", args.llm_threads)
    add_if_present(options, "llm_ctx", args.llm_ctx)
    add_if_present(options, "ocr_rec_batch_num", args.ocr_rec_batch_num)
    add_if_present(options, "ocr_max_side", args.ocr_max_side)
    add_if_present(options, "ocr_scales", args.ocr_scales)
    if args.ocr_use_gpu:
        options["ocr_use_gpu"] = True
    if args.ocr_crop_variants:
        options["ocr_crop_variants"] = True
    if args.ocr_full_image_variants:
        options["ocr_full_image_variants"] = True
    if args.ocr_rotations:
        options["ocr_rotations"] = args.ocr_rotations
    return options


def run_one(
    *,
    base_url: str,
    image: Path,
    image_index: int,
    mode: str,
    args: argparse.Namespace,
    options: dict[str, Any],
    expected: dict[str, Any] | None,
) -> dict[str, Any]:
    endpoint = MODE_ENDPOINTS[mode]
    request_id = f"bench-{mode}-{image_index:04d}-{image.stem[:40]}"
    payload = {
        "request_id": request_id,
        "camera": args.camera,
        "source": "control_server_detection.benchmark",
        "image_b64": image_b64(image),
        "image_format": image_format(image),
        "include_debug": bool(args.include_debug),
        "options": dict(options),
    }
    if args.ocr_rotations:
        payload["ocr_rotations"] = args.ocr_rotations

    started = time.perf_counter()
    status_code, response, raw = http_json(
        "POST",
        f"{base_url}{endpoint}",
        payload=payload,
        timeout=args.timeout,
    )
    latency_s = time.perf_counter() - started
    ok = bool(status_code and 200 <= status_code < 300 and isinstance(response, dict) and response.get("ok", True))
    summary = summarize_response(mode, response)
    evaluation = evaluate(mode, summary, expected)
    error = None
    if not ok:
        error = raw
        if isinstance(response, dict):
            error = response.get("detail") or response.get("error") or raw
    failure_analysis = analyze_failure(
        ok=ok,
        status_code=status_code,
        error=error,
        summary=summary,
        evaluation=evaluation,
        expected=expected,
        response=response,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    row = {
        "image": str(image),
        "image_name": image.name,
        "mode": mode,
        "request_id": request_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "ok": ok,
        "latency_s": latency_s,
        "error": error,
        "summary": summary,
        "evaluation": evaluation,
        "response": response,
    }
    if failure_analysis is not None:
        row["failure_analysis"] = failure_analysis
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_url = (
        os.environ.get("CONTROL_SERVER_DETECTION_SERVICE_URL")
        or os.environ.get("INDORY_OCR_SERVICE_URL")
        or os.environ.get("INDORY_OCR_LLM_SERVICE_URL")
        or "http://127.0.0.1:8767"
    )
    parser = argparse.ArgumentParser(
        description="Benchmark Indory waybill OCR+LLM destination extraction against labeled images.",
    )
    parser.add_argument("inputs", nargs="*", help="image files, directories, or glob patterns")
    parser.add_argument("--url", default=default_url, help="Indory control-server-detection service URL")
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("waybill"))
    parser.add_argument("--out", type=Path, default=Path("benchmark") / "runs" / now_slug())
    parser.add_argument(
        "--dataset",
        help=(
            f"dataset preset ({', '.join(sorted(DATASET_PRESETS))}), manifest/directory path, "
            f"or hf:owner/name; default is {DEFAULT_DATASET!r} when no inputs are provided"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--expected-jsonl", type=Path, help="optional expected labels JSONL")
    parser.add_argument("--recursive", action="store_true", help="scan directories recursively")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of images")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=float(
            os.environ.get(
                "CONTROL_SERVER_DETECTION_BENCH_LOW_CONFIDENCE_THRESHOLD",
                os.environ.get("INDORY_OCR_BENCH_LOW_CONFIDENCE_THRESHOLD", "0.75"),
            )
        ),
        help="decision confidence below this value is reported as low_confidence in failure_analysis",
    )
    parser.add_argument("--camera", default="benchmark")
    parser.add_argument("--option", action="append", default=[], help="provider option as KEY=VALUE; may be repeated")
    parser.add_argument("--ocr-rotations", type=parse_rotations, help="comma-separated rotations, e.g. 0,90,180,270")
    parser.add_argument("--ocr-use-gpu", action="store_true")
    parser.add_argument("--ocr-crop-variants", action="store_true", help="enable waybill crop/upscale OCR variants")
    parser.add_argument("--ocr-full-image-variants", action="store_true", help="enable full-image upscale OCR variants")
    parser.add_argument("--ocr-rec-batch-num", type=int)
    parser.add_argument("--ocr-max-side", type=int)
    parser.add_argument("--ocr-scales", help="semantic OCR scales, e.g. 1.0,2.0")
    parser.add_argument("--judge-mode", choices=["llama_cpp", "openai", "ollama"])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--llm-gpu-layers", type=int)
    parser.add_argument("--llm-threads", type=int)
    parser.add_argument("--llm-ctx", type=int)
    return parser.parse_args(argv)


def print_datasets(dataset_root: Path) -> None:
    print(f"dataset root: {dataset_root}")
    for name, preset in sorted(DATASET_PRESETS.items()):
        source = dataset_root / str(preset["path"])
        expected = dataset_root / str(preset["expected"]) if preset.get("expected") else source
        print(f"{name}:")
        print(f"  source: {source}")
        print(f"  expected: {expected if expected.exists() else '-'}")
        print(f"  description: {preset['description']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_datasets:
        print_datasets(args.dataset_root.expanduser())
        return 0

    base_url = normalize_url(args.url)
    dataset_name = args.dataset
    if not args.inputs and not dataset_name:
        dataset_name = DEFAULT_DATASET

    try:
        dataset_images, dataset_expected, dataset_info = resolve_dataset(
            dataset_name,
            args.dataset_root.expanduser(),
            limit=args.limit,
        )
    except Exception as exc:
        print(f"failed to resolve dataset: {exc}", file=sys.stderr)
        return 2

    images = collect_images(args.inputs, recursive=args.recursive, limit=args.limit) if args.inputs else []
    if dataset_images:
        images = dataset_images + images
        if args.limit > 0:
            images = images[: args.limit]
    if not images:
        print("no benchmark images found", file=sys.stderr)
        return 2

    try:
        expected_map = load_expected(args.expected_jsonl)
    except Exception as exc:
        print(f"failed to load expected labels: {exc}", file=sys.stderr)
        return 2
    expected_map = {**dataset_expected, **expected_map}

    args.out.mkdir(parents=True, exist_ok=True)
    health: dict[str, Any] | None = None
    if not args.skip_health:
        status, payload, raw = http_json("GET", f"{base_url}/health", timeout=args.timeout)
        if status < 200 or status >= 300 or not isinstance(payload, dict):
            print(f"health check failed for {base_url}/health: {raw}", file=sys.stderr)
            return 2
        health = payload

    options = build_mode_options(args)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "service_url": base_url,
        "health": health,
        "dataset": dataset_info,
        "modes": args.modes,
        "image_count": len(images),
        "images": [str(path) for path in images],
        "expected_jsonl": str(args.expected_jsonl) if args.expected_jsonl else None,
        "include_debug": bool(args.include_debug),
        "options": options,
    }
    (args.out / "manifest.json").write_text(json_dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    results_path = args.out / "results.jsonl"
    failures_path = args.out / "failures.jsonl"
    with results_path.open("w", encoding="utf-8") as results_handle, failures_path.open("w", encoding="utf-8") as failures_handle:
        for index, image in enumerate(images, start=1):
            image_expected = expected_for_image(expected_map, image)
            for mode in args.modes:
                row = run_one(
                    base_url=base_url,
                    image=image,
                    image_index=index,
                    mode=mode,
                    args=args,
                    options=options,
                    expected=image_expected,
                )
                rows.append(row)
                results_handle.write(json_dumps(row) + "\n")
                results_handle.flush()
                evaluation = row.get("evaluation") or {}
                if not row.get("ok") or evaluation.get("pass") is False:
                    failures_handle.write(json_dumps(row) + "\n")
                    failures_handle.flush()
                status = "ok" if row.get("ok") else "fail"
                expected_status = ""
                if evaluation.get("evaluated"):
                    expected_status = " expected=pass" if evaluation.get("pass") else " expected=fail"
                print(
                    f"[{index}/{len(images)}] {mode} {image.name}: {status}"
                    f" {row.get('latency_s', 0.0):.2f}s{expected_status}"
                )

    summary = summarize_rows(rows)
    (args.out / "summary.json").write_text(json_dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_md(args.out / "summary.md", summary, manifest)

    print()
    print_table(summary)
    print()
    print(f"wrote: {args.out}")

    if args.fail_on_error:
        any_http_error = any(not row.get("ok") for row in rows)
        any_expected_failure = any((row.get("evaluation") or {}).get("pass") is False for row in rows)
        if any_http_error or any_expected_failure:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
