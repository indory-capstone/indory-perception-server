#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gt
import review


@dataclass(frozen=True)
class ReviewConfig:
    run_dir: Path
    source_manifest: Path
    image_dir: Path
    annotations: Path
    output_manifest: Path
    review_dir: Path
    title: str


WRITE_LOCK = threading.Lock()


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)


def clean_ground_truth(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key in ("destination_floor", "destination_room", "destination_dong"):
        raw = value.get(key)
        if raw not in (None, ""):
            cleaned[key] = str(raw).strip()
    return {key: value for key, value in cleaned.items() if value}


def source_sample_order(source_manifest: Path) -> list[str]:
    _meta, samples = gt.load_manifest_samples(source_manifest)
    return [str(sample.get("sample_id")) for sample in samples if sample.get("sample_id")]


def validate_sample_id(sample_id: str, *, source_manifest: Path) -> None:
    if sample_id not in set(source_sample_order(source_manifest)):
        raise ValueError(f"unknown sample_id: {sample_id}")


def make_annotation_record(payload: dict[str, Any], *, source_manifest: Path) -> dict[str, Any]:
    sample_id = str(payload.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    validate_sample_id(sample_id, source_manifest=source_manifest)

    status = str(payload.get("status") or "verified").strip()
    if status not in gt.ALL_STATUSES:
        raise ValueError(f"unsupported status: {status}")

    ground_truth = clean_ground_truth(payload.get("ground_truth"))
    note = str(payload.get("review_note") or payload.get("note") or "").strip()

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "status": status,
    }
    if status in gt.SCORABLE_STATUSES:
        if not ground_truth:
            raise ValueError("verified status requires at least one ground_truth field")
        record["ground_truth"] = ground_truth
    elif ground_truth:
        record["candidate_ground_truth"] = ground_truth

    if status in gt.EXCLUDE_STATUSES and note:
        record["exclude_reason"] = note
    elif status in gt.UNSCORED_STATUSES and note:
        record["review_reason"] = note
    elif note:
        record["review_note"] = note

    return record


def write_annotations(path: Path, records: dict[str, dict[str, Any]], *, order: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample_id in order:
        if sample_id in records:
            rows.append(records[sample_id])
            seen.add(sample_id)
    for sample_id in sorted(records):
        if sample_id not in seen:
            rows.append(records[sample_id])
    gt.write_jsonl(path, rows)


def rebuild_outputs(config: ReviewConfig) -> dict[str, Any]:
    manifest_summary = gt.build_current_manifest(
        source_manifest=config.source_manifest,
        image_dir=config.image_dir,
        annotations_path=config.annotations,
        output_manifest=config.output_manifest,
        strict_annotations=True,
    )
    review_summary = review.build_review(
        manifest_path=config.output_manifest,
        out_dir=config.review_dir,
        title=config.title,
    )
    return {
        "manifest": manifest_summary,
        "review": review_summary,
    }


def save_annotation(payload: dict[str, Any], *, config: ReviewConfig) -> dict[str, Any]:
    with WRITE_LOCK:
        record = make_annotation_record(payload, source_manifest=config.source_manifest)
        records = gt.load_annotations(config.annotations)
        records[record["sample_id"]] = record
        write_annotations(config.annotations, records, order=source_sample_order(config.source_manifest))
        rebuilt = rebuild_outputs(config)
    return {
        "ok": True,
        "record": record,
        "summary": rebuilt,
    }


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    server_version = "IndoryReviewHTTP/0.1"

    def __init__(self, *args: Any, config: ReviewConfig, **kwargs: Any) -> None:
        self.config = config
        super().__init__(*args, directory=str(config.run_dir), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/review/health":
            self.write_json(
                {
                    "ok": True,
                    "run_dir": str(self.config.run_dir),
                    "manifest": str(self.config.output_manifest),
                    "annotations": str(self.config.annotations),
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/review/save":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown API endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            result = save_annotation(payload, config=self.config)
        except Exception as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self.write_json(result)

    def write_json(self, payload: Any, *, status: int = HTTPStatus.OK) -> None:
        body = (json_dumps(payload, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve editable Indory waybill GT review UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--run-dir", type=Path, default=review.DEFAULT_RUN_DIR)
    parser.add_argument("--source-manifest", type=Path, default=gt.DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=gt.DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=gt.DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-manifest", type=Path, default=gt.DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--review-dir", type=Path, default=review.DEFAULT_OUT_DIR)
    parser.add_argument("--title", default="Indory Waybill GT Review")
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> ReviewConfig:
    return ReviewConfig(
        run_dir=args.run_dir.expanduser().resolve(),
        source_manifest=args.source_manifest.expanduser().resolve(),
        image_dir=args.image_dir.expanduser().resolve(),
        annotations=args.annotations.expanduser().resolve(),
        output_manifest=args.output_manifest.expanduser().resolve(),
        review_dir=args.review_dir.expanduser().resolve(),
        title=args.title,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = make_config(args)
    rebuild_outputs(config)
    handler = functools.partial(ReviewRequestHandler, config=config)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving editable review UI at http://{args.host}:{args.port}/review/index.html", flush=True)
    print(f"Writing annotations to {config.annotations}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
