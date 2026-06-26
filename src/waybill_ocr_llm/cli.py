from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .benchmark import (
    DEFAULT_GENERATIONS_JSONL,
    DEFAULT_GT_DIRS,
    DEFAULT_IMAGE_DIRS,
    DEFAULT_ZIPS,
    benchmark_dataset,
)
from .figures import make_figures
from .llm import judge_ocr_results
from .ocr import run_ocr_on_images
from .pipeline import run_pipeline


DEFAULT_GGUF_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_GGUF_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
FALLBACK_GGUF_FILENAMES = (
    DEFAULT_GGUF_FILENAME,
    "gemma-3-270m-it-Q4_K_M.gguf",
    "gemma_3_270m_it_Q4_K_M.gguf",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_path(mode: str, provided: Path | None) -> Path | None:
    if mode != "llama_cpp":
        return provided

    if provided and provided.is_file():
        return provided

    repo_id = os.environ.get("WAYBILL_OCR_HF_REPO", DEFAULT_GGUF_REPO)
    filename = os.environ.get("WAYBILL_OCR_HF_FILENAME", DEFAULT_GGUF_FILENAME)
    configured_model = (os.environ.get("WAYBILL_OCR_DEFAULT_MODEL", "").strip())

    if configured_model:
        configured_path = Path(configured_model).expanduser()
        if configured_path.is_file():
            return configured_path

    waybill_root = Path(os.environ.get("WAYBILL_OCR_ROOT", str(Path.home() / "waybill_ocr_llm"))).expanduser()
    candidates = [
        waybill_root / "models",
        Path.home() / "waybill_ocr_llm" / "models",
    ]
    if filename:
        for candidate_dir in candidates:
            candidate = candidate_dir / filename
            if candidate.is_file():
                return candidate

    # Fallback names that may be present in old model packs.
    fallback_names = list(dict.fromkeys((filename, *FALLBACK_GGUF_FILENAMES)))
    for filename_fallback in fallback_names:
        for candidate_dir in candidates:
            candidate = candidate_dir / filename_fallback
            if candidate.is_file():
                return candidate

    # Not found in cache; try one-shot HF download with explicit model.
    for candidate_dir in candidates:
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=candidate_dir,
                local_dir_use_symlinks=False,
            )
            return Path(path)
        except Exception:
            continue

    return None


def cmd_run_ocr(args: argparse.Namespace) -> int:
    run_ocr_on_images(
        inputs=args.inputs,
        out_dir=args.out_dir,
        use_gpu=args.ocr_use_gpu,
        rec_batch_num=args.ocr_rec_batch_num,
        limit=args.limit,
        rotations=args.ocr_rotations,
        crop_variants=args.ocr_crop_variants,
        full_image_variants=args.ocr_full_image_variants,
    )
    print(args.out_dir / "waybill_ocr_results.json")
    return 0


def cmd_judge_json(args: argparse.Namespace) -> int:
    ocr_payload = load_json(args.ocr_json)
    endpoint = args.endpoint
    if args.mode == "ollama" and not endpoint:
        endpoint = "http://127.0.0.1:11434/api/chat"
    args.model_path = _resolve_model_path(args.mode, args.model_path)
    judge_ocr_results(
        ocr_payload=ocr_payload,
        mode=args.mode,
        model_path=args.model_path,
        model=args.model,
        endpoint=endpoint,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        chat_format=args.chat_format,
    )
    print(args.out_dir / "waybill_llm_judgements.json")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    endpoint = args.endpoint
    if args.mode == "ollama" and not endpoint:
        endpoint = "http://127.0.0.1:11434/api/chat"
    args.model_path = _resolve_model_path(args.mode, args.model_path)
    run_pipeline(
        inputs=args.inputs,
        out_dir=args.out_dir,
        judge_mode=args.mode,
        model_path=args.model_path,
        model=args.model,
        endpoint=endpoint,
        ocr_use_gpu=args.ocr_use_gpu,
        ocr_rec_batch_num=args.ocr_rec_batch_num,
        ocr_rotations=args.ocr_rotations,
        ocr_crop_variants=args.ocr_crop_variants,
        ocr_full_image_variants=args.ocr_full_image_variants,
        llm_gpu_layers=args.n_gpu_layers,
        llm_threads=args.n_threads,
        llm_ctx=args.n_ctx,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(args.out_dir / "waybill_pipeline_results.json")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    make_figures(args.ocr_json, args.llm_json, args.out_dir, eval_csv=args.eval_csv)
    print(args.out_dir / "ocr_llm_contact_sheet.jpg")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    endpoint = args.endpoint
    if args.mode == "ollama" and not endpoint:
        endpoint = "http://127.0.0.1:11434/api/chat"
    if args.run_pipeline:
        args.model_path = _resolve_model_path(args.mode, args.model_path)
    result = benchmark_dataset(
        out_dir=args.out_dir,
        image_dirs=args.image_dir,
        zip_paths=args.zip,
        gt_dirs=args.gt_dir,
        generations_jsonl=args.generations_jsonl,
        run_pipeline_enabled=args.run_pipeline,
        pipeline_json=args.pipeline_json,
        judge_mode=args.mode,
        model_path=args.model_path,
        model=args.model,
        endpoint=endpoint,
        ocr_use_gpu=args.ocr_use_gpu,
        ocr_rec_batch_num=args.ocr_rec_batch_num,
        ocr_full_image_variants=args.ocr_full_image_variants,
        llm_gpu_layers=args.n_gpu_layers,
        llm_threads=args.n_threads,
        llm_ctx=args.n_ctx,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        include_qc_fail=args.include_qc_fail,
        include_label_only=args.include_label_only,
        resize=parse_resize(args.resize) if args.resize else None,
        resize_mode=args.resize_mode,
        resize_jpeg_quality=args.resize_jpeg_quality,
    )
    print(result["report"])
    return 0


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["llama_cpp", "openai", "ollama"], default="llama_cpp")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--chat-format", default="")


def parse_resize(value: str) -> tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resize must be WIDTHxHEIGHT, for example 640x480")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resize must be WIDTHxHEIGHT, for example 640x480") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resize dimensions must be positive")
    return width, height


def parse_ocr_rotations(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = [part.strip() for part in str(value).split(",")]
    rotations: list[int] = []
    for part in parts:
        if not part:
            continue
        try:
            rotation = int(part) % 360
        except ValueError as exc:
            raise argparse.ArgumentTypeError("ocr rotations must be comma-separated degrees, for example 0,90,180,270") from exc
        if rotation not in {0, 90, 180, 270}:
            raise argparse.ArgumentTypeError("ocr rotations must use only 0,90,180,270")
        if rotation not in rotations:
            rotations.append(rotation)
    return rotations or [0]


def main() -> int:
    parser = argparse.ArgumentParser(prog="waybill-ocr-llm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_ocr = sub.add_parser("run-ocr", help="run PaddleOCR and write waybill_ocr_results.json")
    run_ocr.add_argument("inputs", nargs="+", type=Path)
    run_ocr.add_argument("--out-dir", type=Path, required=True)
    run_ocr.add_argument("--limit", type=int, default=0)
    run_ocr.add_argument("--ocr-use-gpu", action="store_true")
    run_ocr.add_argument("--ocr-rec-batch-num", type=int, default=1)
    run_ocr.add_argument("--ocr-rotations", type=parse_ocr_rotations, default=parse_ocr_rotations(os.environ.get("WAYBILL_OCR_ROTATIONS", "0")), help="comma-separated rotation variants, for example 0,90,180,270")
    run_ocr.add_argument("--ocr-crop-variants", action="store_true", default=os.environ.get("WAYBILL_OCR_CROP_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"})
    run_ocr.add_argument("--ocr-full-image-variants", action="store_true", default=os.environ.get("WAYBILL_OCR_FULL_IMAGE_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"})
    run_ocr.set_defaults(func=cmd_run_ocr)

    judge = sub.add_parser("judge-json", help="run LLM candidate selection on an OCR JSON")
    judge.add_argument("ocr_json", type=Path)
    judge.add_argument("--out-dir", type=Path, required=True)
    add_judge_args(judge)
    judge.set_defaults(func=cmd_judge_json)

    run_all = sub.add_parser("run", help="run OCR and LLM candidate selection")
    run_all.add_argument("inputs", nargs="+", type=Path)
    run_all.add_argument("--out-dir", type=Path, required=True)
    run_all.add_argument("--limit", type=int, default=0)
    run_all.add_argument("--ocr-use-gpu", action="store_true")
    run_all.add_argument("--ocr-rec-batch-num", type=int, default=1)
    run_all.add_argument("--ocr-rotations", type=parse_ocr_rotations, default=parse_ocr_rotations(os.environ.get("WAYBILL_OCR_ROTATIONS", "0")), help="comma-separated rotation variants, for example 0,90,180,270")
    run_all.add_argument("--ocr-crop-variants", action="store_true", default=os.environ.get("WAYBILL_OCR_CROP_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"})
    run_all.add_argument("--ocr-full-image-variants", action="store_true", default=os.environ.get("WAYBILL_OCR_FULL_IMAGE_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"})
    add_judge_args(run_all)
    run_all.set_defaults(func=cmd_run)

    figs = sub.add_parser("figures", help="make OCR/LLM evidence figures")
    figs.add_argument("--ocr-json", type=Path, required=True)
    figs.add_argument("--llm-json", type=Path, required=True)
    figs.add_argument("--eval-csv", type=Path)
    figs.add_argument("--out-dir", type=Path, required=True)
    figs.set_defaults(func=cmd_figures)

    bench = sub.add_parser("benchmark", help="build/evaluate a waybill OCR+LLM benchmark dataset")
    bench.add_argument("--out-dir", type=Path, required=True)
    bench.add_argument("--image-dir", type=Path, action="append", default=list(DEFAULT_IMAGE_DIRS))
    bench.add_argument("--zip", type=Path, action="append", default=list(DEFAULT_ZIPS))
    bench.add_argument("--gt-dir", type=Path, action="append", default=list(DEFAULT_GT_DIRS))
    bench.add_argument("--generations-jsonl", type=Path, default=DEFAULT_GENERATIONS_JSONL)
    bench.add_argument("--pipeline-json", type=Path, help="existing waybill_pipeline_results.json to evaluate")
    bench.add_argument("--run-pipeline", action="store_true", help="run OCR+LLM before computing metrics")
    bench.add_argument("--include-qc-fail", action="store_true", help="include QC-failed synthetic samples in labeled metrics")
    bench.add_argument("--include-label-only", action="store_true", help="include standalone waybill-label renderings; default benchmark uses box-scene images only")
    bench.add_argument("--limit", type=int, default=0)
    bench.add_argument("--resize", default="", help="resize every benchmark input to WIDTHxHEIGHT, for example 640x480")
    bench.add_argument("--resize-mode", choices=["contain", "cover", "stretch"], default="contain")
    bench.add_argument("--resize-jpeg-quality", type=int, default=85)
    bench.add_argument("--ocr-use-gpu", action="store_true")
    bench.add_argument("--ocr-rec-batch-num", type=int, default=1)
    bench.add_argument("--ocr-full-image-variants", action="store_true", default=os.environ.get("WAYBILL_OCR_FULL_IMAGE_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"})
    add_judge_args(bench)
    bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
