from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from .llm import infer_floor_from_room, normalize_dong, normalize_floor, normalize_room
from .ocr import IMAGE_SUFFIXES, image_inputs
from .pipeline import run_pipeline


def _env_path(name: str, fallback: Path) -> Path:
    return Path(os.environ.get(name, str(fallback))).expanduser()


def _env_path_list(name: str, fallback: list[Path]) -> list[Path]:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


DEFAULT_DATA_ROOT = _env_path("WAYBILL_BENCHMARK_DATA_ROOT", Path.home() / "data")
DEFAULT_IMAGE_DIRS = _env_path_list(
    "WAYBILL_BENCHMARK_IMAGE_DIRS",
    [
        DEFAULT_DATA_ROOT / "generated" / "labels" / "all",
        DEFAULT_DATA_ROOT / "generated" / "labels" / "previews",
    ],
)
DEFAULT_ZIPS = _env_path_list(
    "WAYBILL_BENCHMARK_ZIPS",
    [DEFAULT_DATA_ROOT / "generated" / "waybill.zip"],
)
DEFAULT_GT_DIRS = _env_path_list(
    "WAYBILL_BENCHMARK_GT_DIRS",
    [
        DEFAULT_DATA_ROOT / "metadata" / "ground_truth",
        DEFAULT_DATA_ROOT / "metadata" / "label_ground_truth",
    ],
)
DEFAULT_GENERATIONS_JSONL = _env_path(
    "WAYBILL_BENCHMARK_GENERATIONS_JSONL",
    DEFAULT_DATA_ROOT / "metadata" / "generations.jsonl",
)

KNOWN_CARRIERS = ("cj_daehan", "coupang_cls", "korea_post", "hanjin", "logen", "lotte")
KNOWN_CONDITIONS = (
    "front_clean",
    "bright_light",
    "low_light",
    "oblique_angle",
    "partial_occlusion",
    "slight_wrinkle",
    "combined_hard",
)

DESTINATION_FIELDS = ("destination_room", "destination_floor", "destination_dong")
OCR_REFERENCE_FIELDS = (
    "recipient_name",
    "recipient_address_base",
    "recipient_detail_address",
    "destination_code",
    "tracking_code_masked",
    "order_code_masked",
)


def current_vram_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    values = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
    return max(values) if values else None


class GpuMonitor:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = current_vram_mb()
            if value is not None:
                self.samples.append((time.perf_counter(), value))
            time.sleep(self.interval)

    @contextmanager
    def span(self) -> Iterator[dict[str, Any]]:
        start_time = time.perf_counter()
        baseline = current_vram_mb()
        record: dict[str, Any] = {"start_time": start_time, "baseline_vram_mb": baseline}
        try:
            yield record
        finally:
            end_time = time.perf_counter()
            record["end_time"] = end_time
            samples = [v for t, v in self.samples if start_time <= t <= end_time]
            if not samples and baseline is not None:
                samples = [baseline]
            record["mean_vram_mb"] = statistics.mean(samples) if samples else None
            record["peak_vram_mb"] = max(samples) if samples else None
            if baseline is None or not samples:
                record["mean_vram_delta_mb"] = None
                record["peak_vram_delta_mb"] = None
            else:
                record["mean_vram_delta_mb"] = max(0.0, float(statistics.mean(samples) - baseline))
                record["peak_vram_delta_mb"] = max(0, int(max(samples) - baseline))


def summarize_times(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "total_sec": 0.0, "mean_sec": None, "median_sec": None, "min_sec": None, "max_sec": None, "fps": None}
    total = sum(values)
    return {
        "count": len(values),
        "total_sec": total,
        "mean_sec": statistics.mean(values),
        "median_sec": statistics.median(values),
        "min_sec": min(values),
        "max_sec": max(values),
        "fps": len(values) / total if total > 0 else None,
    }


def write_benchmark_summary(benchmark_json: Path, out_md: Path) -> None:
    data = json.loads(benchmark_json.read_text(encoding="utf-8"))
    lines = [
        "# Waybill OCR LLM Benchmark",
        "",
        f"Images: {data.get('image_count')}",
        f"OCR model: {data.get('ocr_model')}",
        f"LLM model: {data.get('model_path')}",
        "",
        "| Component | Total sec | Mean sec | Median sec | Min sec | Max sec | FPS | Mean raw VRAM MB | Peak raw VRAM MB | Mean delta MB | Peak delta MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, component in data.get("components", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(component.get("total_sec") or component.get("seconds")),
                    _fmt(component.get("mean_sec") or component.get("seconds")),
                    _fmt(component.get("median_sec")),
                    _fmt(component.get("min_sec")),
                    _fmt(component.get("max_sec")),
                    _fmt(component.get("fps")),
                    _fmt(component.get("mean_vram_mb")),
                    _fmt(component.get("peak_vram_mb")),
                    _fmt(component.get("mean_vram_delta_mb")),
                    _fmt(component.get("peak_vram_delta_mb")),
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def benchmark_dataset(
    out_dir: Path,
    image_dirs: list[Path] | None = None,
    zip_paths: list[Path] | None = None,
    gt_dirs: list[Path] | None = None,
    generations_jsonl: Path | None = DEFAULT_GENERATIONS_JSONL,
    run_pipeline_enabled: bool = False,
    pipeline_json: Path | None = None,
    judge_mode: str = "llama_cpp",
    model_path: Path | None = None,
    model: str = "",
    endpoint: str = "",
    ocr_use_gpu: bool = False,
    ocr_rec_batch_num: int = 1,
    ocr_full_image_variants: bool | None = None,
    llm_gpu_layers: int = 0,
    llm_threads: int = 4,
    llm_ctx: int = 1024,
    max_new_tokens: int = 64,
    limit: int = 0,
    include_qc_fail: bool = False,
    include_label_only: bool = False,
    resize: tuple[int, int] | None = None,
    resize_mode: str = "contain",
    resize_jpeg_quality: int = 85,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = image_dirs if image_dirs is not None else DEFAULT_IMAGE_DIRS
    zip_paths = zip_paths if zip_paths is not None else DEFAULT_ZIPS
    gt_dirs = gt_dirs if gt_dirs is not None else DEFAULT_GT_DIRS

    generation_meta = load_generation_metadata(generations_jsonl)
    samples = collect_samples(
        out_dir=out_dir,
        image_dirs=image_dirs,
        zip_paths=zip_paths,
        gt_dirs=gt_dirs,
        generation_meta=generation_meta,
        include_qc_fail=include_qc_fail,
        include_label_only=include_label_only,
        resize=resize,
        resize_mode=resize_mode,
        resize_jpeg_quality=resize_jpeg_quality,
    )
    if limit > 0:
        samples = samples[:limit]

    manifest = {
        "created_at_unix": time.time(),
        "image_dirs": [str(path) for path in image_dirs],
        "zip_paths": [str(path) for path in zip_paths],
        "gt_dirs": [str(path) for path in gt_dirs],
        "generations_jsonl": str(generations_jsonl) if generations_jsonl else None,
        "include_qc_fail": include_qc_fail,
        "include_label_only": include_label_only,
        "resize": {"width": resize[0], "height": resize[1], "mode": resize_mode, "jpeg_quality": resize_jpeg_quality}
        if resize
        else None,
        "samples": samples,
    }
    (out_dir / "benchmark_manifest.json").write_text(json_dumps(manifest), encoding="utf-8")

    summary = summarize_dataset(samples)
    (out_dir / "dataset_summary.json").write_text(json_dumps(summary), encoding="utf-8")

    pipeline_payload: dict[str, Any] | None = None
    if pipeline_json is not None:
        pipeline_payload = json.loads(pipeline_json.read_text(encoding="utf-8"))
    elif run_pipeline_enabled:
        run_images = [Path(sample["image_path"]) for sample in samples]
        pipeline_payload = run_pipeline(
            inputs=run_images,
            out_dir=out_dir / "pipeline",
            judge_mode=judge_mode,
            model_path=model_path,
            model=model,
            endpoint=endpoint,
            ocr_use_gpu=ocr_use_gpu,
            ocr_rec_batch_num=ocr_rec_batch_num,
            ocr_full_image_variants=ocr_full_image_variants,
            llm_gpu_layers=llm_gpu_layers,
            llm_threads=llm_threads,
            llm_ctx=llm_ctx,
            max_new_tokens=max_new_tokens,
        )

    metrics: dict[str, Any] | None = None
    if pipeline_payload is not None:
        metrics = evaluate_pipeline(samples=samples, pipeline_payload=pipeline_payload, out_dir=out_dir)
        (out_dir / "evaluation_metrics.json").write_text(json_dumps(metrics), encoding="utf-8")

    report = render_report(summary=summary, metrics=metrics)
    (out_dir / "benchmark_report.md").write_text(report, encoding="utf-8")

    return {
        "manifest": str(out_dir / "benchmark_manifest.json"),
        "dataset_summary": str(out_dir / "dataset_summary.json"),
        "evaluation_metrics": str(out_dir / "evaluation_metrics.json") if metrics else None,
        "report": str(out_dir / "benchmark_report.md"),
        "summary": summary,
        "metrics": metrics,
    }


def collect_samples(
    out_dir: Path,
    image_dirs: list[Path],
    zip_paths: list[Path],
    gt_dirs: list[Path],
    generation_meta: dict[str, dict[str, Any]],
    include_qc_fail: bool,
    include_label_only: bool,
    resize: tuple[int, int] | None,
    resize_mode: str,
    resize_jpeg_quality: int,
) -> list[dict[str, Any]]:
    samples_by_id: dict[str, dict[str, Any]] = {}

    def add_image(path: Path, source: str, source_container: str | None = None) -> None:
        sample_id = path.stem
        input_kind = infer_input_kind(path, source)
        if input_kind == "label_only" and not include_label_only:
            return
        image_path = path
        original_size = image_size(path)
        if resize is not None:
            image_path = resize_image(
                source_path=path,
                out_path=out_dir / "resized_images" / f"{sample_id}_{resize[0]}x{resize[1]}.jpg",
                size=resize,
                mode=resize_mode,
                jpeg_quality=resize_jpeg_quality,
            )
        gt_path, gt_payload = find_ground_truth(sample_id, gt_dirs)
        meta = generation_meta.get(sample_id, {})
        qc_status = str(meta.get("qc_status") or "")
        qc_grade = str(meta.get("qc_grade") or "")
        qc_failed = qc_grade == "fail" or qc_status.startswith("rejected")
        labeled = gt_payload is not None and (include_qc_fail or not qc_failed)
        entry = {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "original_image_path": str(path),
            "original_size": original_size,
            "input_size": {"width": resize[0], "height": resize[1]} if resize else original_size,
            "resize_mode": resize_mode if resize else None,
            "source": source,
            "source_container": source_container,
            "input_kind": input_kind,
            "carrier": infer_carrier(sample_id, gt_payload, meta),
            "condition": infer_condition(sample_id, gt_payload, meta),
            "seed_id": infer_seed_id(sample_id, gt_payload, meta),
            "gt_path": str(gt_path) if gt_path else None,
            "has_ground_truth": gt_payload is not None,
            "is_labeled_eval": labeled,
            "exclude_reason": "qc_failed_generation" if gt_payload is not None and qc_failed and not include_qc_fail else None,
            "qc_status": qc_status or None,
            "qc_grade": qc_grade or None,
            "ground_truth": normalize_ground_truth(gt_payload) if gt_payload else None,
        }

        existing = samples_by_id.get(sample_id)
        if existing is None or source_priority(source) < source_priority(str(existing.get("source") or "")):
            if existing is not None:
                entry["alternate_paths"] = sorted(set(existing.get("alternate_paths", []) + [existing["original_image_path"]]))
            samples_by_id[sample_id] = entry
        else:
            existing.setdefault("alternate_paths", [])
            existing["alternate_paths"] = sorted(set(existing["alternate_paths"] + [str(path)]))

    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for image in image_inputs([image_dir]):
            add_image(image, source=f"dir:{image_dir}")

    extract_root = out_dir / "extracted_zip_images"
    for zip_path in zip_paths:
        if not zip_path.exists():
            continue
        target_root = extract_root / zip_path.stem
        target_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                member_path = Path(member)
                if member.startswith("__MACOSX/") or member_path.name.startswith("._"):
                    continue
                if member_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                out_path = target_root / member_path.name
                if not out_path.exists():
                    out_path.write_bytes(zf.read(member))
                add_image(out_path, source=f"zip:{zip_path}", source_container=member)

    return sorted(samples_by_id.values(), key=lambda sample: sample["sample_id"])


def image_size(path: Path) -> dict[str, int] | None:
    try:
        with Image.open(path) as image:
            return {"width": int(image.width), "height": int(image.height)}
    except Exception:
        return None


def resize_image(
    source_path: Path,
    out_path: Path,
    size: tuple[int, int],
    mode: str,
    jpeg_quality: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    width, height = size
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if mode == "stretch":
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
        elif mode == "cover":
            scale = max(width / image.width, height / image.height)
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            scaled = image.resize(new_size, Image.Resampling.LANCZOS)
            left = max(0, (scaled.width - width) // 2)
            top = max(0, (scaled.height - height) // 2)
            resized = scaled.crop((left, top, left + width, top + height))
        elif mode == "contain":
            scale = min(width / image.width, height / image.height)
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            scaled = image.resize(new_size, Image.Resampling.LANCZOS)
            resized = Image.new("RGB", (width, height), (0, 0, 0))
            resized.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
        else:
            raise ValueError(f"Unsupported resize mode: {mode}")
        resized.save(out_path, format="JPEG", quality=max(1, min(100, jpeg_quality)), optimize=True)
    return out_path


def source_priority(source: str) -> int:
    if source.startswith("zip:"):
        return 0
    if source.startswith("dir:") and "/all" in source:
        return 1
    if source.startswith("dir:") and "/previews" in source:
        return 2
    return 3


def infer_input_kind(path: Path, source: str) -> str:
    if source.startswith("zip:"):
        return "box_scene"
    path_text = str(path)
    if "/generated/labels/" in path_text or "/labels/all/" in path_text or "/labels/previews/" in path_text:
        return "label_only"
    return "box_scene"


def find_ground_truth(sample_id: str, gt_dirs: list[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    for gt_dir in gt_dirs:
        path = gt_dir / f"{sample_id}.json"
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, None


def load_generation_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        sample_id = payload.get("sample_id")
        if isinstance(sample_id, str):
            records[sample_id] = payload
    return records


def normalize_ground_truth(payload: dict[str, Any]) -> dict[str, Any]:
    recipient = payload.get("recipient") if isinstance(payload.get("recipient"), dict) else {}
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    recipient_name = first_text(recipient.get("name"), payload.get("recipient_name"))
    address_base = first_text(recipient.get("address_base"), payload.get("recipient_address_base"))
    detail_address = first_text(recipient.get("detail_address"), payload.get("recipient_detail_address"))
    destination_code = first_text(payload.get("destination_code"))
    combined_address = " ".join(part for part in (address_base, detail_address) if part)
    destination_room = normalize_room(detail_address) or normalize_room(combined_address)
    destination_dong = normalize_dong(detail_address) or normalize_dong(combined_address)
    destination_floor = normalize_floor(payload.get("destination_floor")) or infer_floor_from_room(destination_room)

    return {
        "sample_id": payload.get("sample_id") or payload.get("label_id"),
        "carrier": payload.get("carrier"),
        "condition": payload.get("condition"),
        "seed_id": payload.get("seed_id"),
        "recipient_name": recipient_name,
        "recipient_address_base": address_base,
        "recipient_detail_address": detail_address,
        "recipient_phone_masked": first_text(recipient.get("phone_masked"), payload.get("recipient_phone_masked")),
        "sender_name": first_text(sender.get("name"), payload.get("sender_name")),
        "destination_code": destination_code,
        "tracking_code_masked": first_text(payload.get("tracking_code_masked")),
        "order_code_masked": first_text(payload.get("order_code_masked")),
        "destination_room": destination_room,
        "destination_floor": destination_floor,
        "destination_dong": destination_dong,
    }


def first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def infer_carrier(sample_id: str, gt_payload: dict[str, Any] | None, meta: dict[str, Any]) -> str:
    gt_carrier = gt_payload.get("carrier") if gt_payload else None
    if isinstance(gt_carrier, str) and gt_carrier:
        return gt_carrier
    meta_carrier = meta.get("carrier")
    if isinstance(meta_carrier, str) and meta_carrier:
        return meta_carrier
    for carrier in KNOWN_CARRIERS:
        if sample_id.startswith(f"{carrier}_"):
            return carrier
    return "unknown"


def infer_condition(sample_id: str, gt_payload: dict[str, Any] | None, meta: dict[str, Any]) -> str:
    gt_condition = gt_payload.get("condition") if gt_payload else None
    if isinstance(gt_condition, str) and gt_condition:
        return gt_condition
    meta_condition = meta.get("condition")
    if isinstance(meta_condition, str) and meta_condition:
        return meta_condition
    for condition in KNOWN_CONDITIONS:
        if condition in sample_id:
            return condition
    return "unknown"


def infer_seed_id(sample_id: str, gt_payload: dict[str, Any] | None, meta: dict[str, Any]) -> str | None:
    gt_seed = gt_payload.get("seed_id") if gt_payload else None
    if isinstance(gt_seed, str) and gt_seed:
        return gt_seed
    meta_seed = meta.get("seed_id")
    if isinstance(meta_seed, str) and meta_seed:
        return meta_seed
    parts = sample_id.split("_")
    for idx, part in enumerate(parts[:-1]):
        if part == "seed" and parts[idx + 1].isdigit():
            return f"seed_{parts[idx + 1]}"
    return None


def summarize_dataset(samples: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [sample for sample in samples if sample.get("is_labeled_eval")]
    unlabeled = [sample for sample in samples if not sample.get("has_ground_truth")]
    excluded = [sample for sample in samples if sample.get("exclude_reason")]
    return {
        "total_images": len(samples),
        "labeled_eval_images": len(labeled),
        "unlabeled_images": len(unlabeled),
        "excluded_labeled_images": len(excluded),
        "by_source": count_by(samples, "source"),
        "by_input_kind": count_by(samples, "input_kind"),
        "by_carrier": count_by(samples, "carrier"),
        "by_condition": count_by(samples, "condition"),
        "labeled_by_carrier": count_by(labeled, "carrier"),
        "labeled_by_condition": count_by(labeled, "condition"),
        "unlabeled_examples": [sample["sample_id"] for sample in unlabeled[:20]],
        "excluded_examples": [
            {"sample_id": sample["sample_id"], "reason": sample.get("exclude_reason")}
            for sample in excluded[:20]
        ],
    }


def count_by(samples: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        value = str(sample.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_pipeline(samples: list[dict[str, Any]], pipeline_payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    ocr_by_id = {
        sample_id_from_image_path(result.get("image")): result
        for result in (pipeline_payload.get("ocr") or {}).get("results") or []
        if isinstance(result, dict)
    }
    llm_by_id = {
        sample_id_from_image_path(result.get("image")): result
        for result in (pipeline_payload.get("llm") or {}).get("results") or []
        if isinstance(result, dict)
    }
    pipeline_sample_ids = set(ocr_by_id) | set(llm_by_id)
    if pipeline_sample_ids:
        samples = [sample for sample in samples if sample.get("sample_id") in pipeline_sample_ids]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        gt = sample.get("ground_truth") or {}
        ocr_result = ocr_by_id.get(sample_id, {})
        llm_result = llm_by_id.get(sample_id, {})
        decision = llm_result.get("llm_decision") or {}
        candidates = llm_result.get("destination_candidates") or []
        ocr_text = str(llm_result.get("full_ocr_text") or " ".join(item.get("text", "") for item in ocr_result.get("ocr_items") or []))

        row = {
            "sample_id": sample_id,
            "input_kind": sample.get("input_kind"),
            "carrier": sample.get("carrier"),
            "condition": sample.get("condition"),
            "is_labeled_eval": bool(sample.get("is_labeled_eval")),
            "ocr_item_count": int(ocr_result.get("ocr_item_count") or 0),
            "has_ocr_text": bool(ocr_result.get("ocr_item_count") or ocr_text.strip()),
            "candidate_count": len(candidates),
            "pred_destination_label": llm_result.get("llm_destination_label"),
            "pred_room": decision.get("destination_room"),
            "pred_floor": decision.get("destination_floor"),
            "pred_dong": decision.get("destination_dong"),
            "gt_room": gt.get("destination_room"),
            "gt_floor": gt.get("destination_floor"),
            "gt_dong": gt.get("destination_dong"),
            "destination_exact": None,
            "room_match": None,
            "floor_match": None,
            "dong_match": None,
            "candidate_recall": None,
            "ocr_field_exact_rate": None,
            "ocr_field_char_recall": None,
        }

        if sample.get("is_labeled_eval"):
            row["room_match"] = nullable_match(gt.get("destination_room"), decision.get("destination_room"))
            row["floor_match"] = nullable_match(gt.get("destination_floor"), decision.get("destination_floor"))
            row["dong_match"] = nullable_match(gt.get("destination_dong"), decision.get("destination_dong"))
            row["destination_exact"] = destination_exact(gt, decision)
            row["candidate_recall"] = candidate_recall(gt, candidates)
            ocr_scores = score_ocr_reference_fields(gt, ocr_text)
            row["ocr_field_exact_rate"] = ocr_scores["exact_rate"]
            row["ocr_field_char_recall"] = ocr_scores["char_recall_mean"]
            if not row["destination_exact"]:
                failures.append(row)

        rows.append(row)

    write_csv(out_dir / "evaluation_rows.csv", rows)
    write_csv(out_dir / "failures.csv", failures)

    labeled_rows = [row for row in rows if row["is_labeled_eval"]]
    metrics = {
        "total_images": len(rows),
        "evaluated_labeled_images": len(labeled_rows),
        "unlabeled_images": len(rows) - len(labeled_rows),
        "ocr_text_detection_rate": mean_bool(row["has_ocr_text"] for row in rows),
        "destination_exact_accuracy": mean_bool(row["destination_exact"] for row in labeled_rows),
        "room_accuracy": mean_bool(row["room_match"] for row in labeled_rows if row["room_match"] is not None),
        "floor_accuracy": mean_bool(row["floor_match"] for row in labeled_rows if row["floor_match"] is not None),
        "dong_accuracy": mean_bool(row["dong_match"] for row in labeled_rows if row["dong_match"] is not None),
        "candidate_recall": mean_bool(row["candidate_recall"] for row in labeled_rows if row["candidate_recall"] is not None),
        "ocr_field_exact_rate": mean_float(row["ocr_field_exact_rate"] for row in labeled_rows),
        "ocr_field_char_recall": mean_float(row["ocr_field_char_recall"] for row in labeled_rows),
        "failure_count": len(failures),
        "by_carrier": grouped_metrics(labeled_rows, "carrier"),
        "by_condition": grouped_metrics(labeled_rows, "condition"),
    }
    return metrics


def sample_id_from_image_path(value: Any) -> str:
    stem = Path(str(value or "")).stem
    return re.sub(r"_\d+x\d+$", "", stem)


def nullable_match(expected: Any, actual: Any) -> bool | None:
    if expected is None:
        return None
    return expected == actual


def destination_exact(gt: dict[str, Any], decision: dict[str, Any]) -> bool | None:
    expected_fields = [field for field in DESTINATION_FIELDS if gt.get(field) is not None]
    if not expected_fields:
        return None
    return all(gt.get(field) == decision.get(field) for field in expected_fields)


def candidate_recall(gt: dict[str, Any], candidates: list[dict[str, Any]]) -> bool | None:
    expected_fields = [field for field in DESTINATION_FIELDS if gt.get(field) is not None]
    if not expected_fields:
        return None
    for candidate in candidates:
        normalized = {
            "destination_room": normalize_room(candidate.get("destination_room")),
            "destination_floor": normalize_floor(candidate.get("destination_floor")),
            "destination_dong": normalize_dong(candidate.get("destination_dong")),
        }
        if normalized["destination_floor"] is None:
            normalized["destination_floor"] = infer_floor_from_room(normalized["destination_room"])
        if all(gt.get(field) == normalized.get(field) for field in expected_fields):
            return True
    return False


def score_ocr_reference_fields(gt: dict[str, Any], ocr_text: str) -> dict[str, float | int | None]:
    expected_values = [str(gt[field]) for field in OCR_REFERENCE_FIELDS if gt.get(field)]
    if not expected_values:
        return {"field_count": 0, "exact_rate": None, "char_recall_mean": None}
    exact_hits = 0
    recalls: list[float] = []
    normalized_ocr = normalize_eval_text(ocr_text)
    for expected in expected_values:
        normalized_expected = normalize_eval_text(expected)
        if not normalized_expected:
            continue
        exact = normalized_expected in normalized_ocr
        exact_hits += int(exact)
        recalls.append(sequence_recall(normalized_expected, normalized_ocr))
    return {
        "field_count": len(expected_values),
        "exact_rate": exact_hits / len(expected_values) if expected_values else None,
        "char_recall_mean": statistics.mean(recalls) if recalls else None,
    }


def normalize_eval_text(text: str) -> str:
    keep = []
    for char in str(text).lower():
        if char.isalnum() or "\uac00" <= char <= "\ud7a3":
            keep.append(char)
    return "".join(keep)


def sequence_recall(expected: str, actual: str) -> float:
    if not expected:
        return 0.0
    from difflib import SequenceMatcher

    blocks = SequenceMatcher(None, expected, actual).get_matching_blocks()
    return sum(block.size for block in blocks) / len(expected)


def mean_bool(values: Iterable[Any]) -> float | None:
    parsed = [bool(value) for value in values if value is not None]
    return statistics.mean(parsed) if parsed else None


def mean_float(values: Iterable[Any]) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    return statistics.mean(parsed) if parsed else None


def grouped_metrics(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key) or "unknown"), []).append(row)
    return {
        group: {
            "count": len(group_rows),
            "destination_exact_accuracy": mean_bool(row["destination_exact"] for row in group_rows),
            "candidate_recall": mean_bool(row["candidate_recall"] for row in group_rows if row["candidate_recall"] is not None),
            "ocr_field_char_recall": mean_float(row["ocr_field_char_recall"] for row in group_rows),
            "failure_count": sum(1 for row in group_rows if row["destination_exact"] is False),
        }
        for group, group_rows in sorted(grouped.items())
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    lines = [
        "# Waybill OCR+LLM Benchmark Report",
        "",
        "## Dataset",
        "",
        f"- Total images: {summary['total_images']}",
        f"- Labeled evaluation images: {summary['labeled_eval_images']}",
        f"- Unlabeled stress/generalization images: {summary['unlabeled_images']}",
        f"- Excluded labeled images: {summary['excluded_labeled_images']}",
        "",
        "### Images by Input Kind",
        "",
        markdown_table(summary.get("by_input_kind") or {}, ["Input Kind", "Images"]),
        "",
        "### Labeled Images by Carrier",
        "",
        markdown_table(summary["labeled_by_carrier"], ["Carrier", "Images"]),
        "",
        "### Labeled Images by Condition",
        "",
        markdown_table(summary["labeled_by_condition"], ["Condition", "Images"]),
        "",
    ]
    if metrics is None:
        lines.extend(
            [
                "## Metrics",
                "",
                "Pipeline results were not supplied yet. Run with `--run-pipeline` or pass `--pipeline-json` to compute quantitative OCR/LLM metrics.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| OCR text detection rate | {format_percent(metrics.get('ocr_text_detection_rate'))} |",
            f"| Destination exact accuracy | {format_percent(metrics.get('destination_exact_accuracy'))} |",
            f"| Destination candidate recall | {format_percent(metrics.get('candidate_recall'))} |",
            f"| Room accuracy | {format_percent(metrics.get('room_accuracy'))} |",
            f"| Floor accuracy | {format_percent(metrics.get('floor_accuracy'))} |",
            f"| Dong accuracy | {format_percent(metrics.get('dong_accuracy'))} |",
            f"| OCR field exact hit rate | {format_percent(metrics.get('ocr_field_exact_rate'))} |",
            f"| OCR field character recall | {format_percent(metrics.get('ocr_field_char_recall'))} |",
            f"| Failure count | {metrics.get('failure_count')} |",
            "",
            "### By Carrier",
            "",
            grouped_markdown_table(metrics.get("by_carrier") or {}, "Carrier"),
            "",
            "### By Condition",
            "",
            grouped_markdown_table(metrics.get("by_condition") or {}, "Condition"),
            "",
            "Failure details are written to `failures.csv`; per-image rows are written to `evaluation_rows.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_table(counts: dict[str, int], headers: list[str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | ---: |"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def grouped_markdown_table(groups: dict[str, dict[str, Any]], label: str) -> str:
    lines = [
        f"| {label} | Images | Destination exact | Candidate recall | OCR char recall | Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, values in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(values.get("count")),
                    format_percent(values.get("destination_exact_accuracy")),
                    format_percent(values.get("candidate_recall")),
                    format_percent(values.get("ocr_field_char_recall")),
                    str(values.get("failure_count")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
