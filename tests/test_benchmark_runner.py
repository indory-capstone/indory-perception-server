from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
SPEC = importlib.util.spec_from_file_location("indory_ocr_benchmark_run", ROOT / "benchmark" / "run.py")
assert SPEC is not None
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench)

GT_SPEC = importlib.util.spec_from_file_location("indory_ocr_benchmark_gt", ROOT / "benchmark" / "gt.py")
assert GT_SPEC is not None
gt = importlib.util.module_from_spec(GT_SPEC)
assert GT_SPEC.loader is not None
GT_SPEC.loader.exec_module(gt)

REVIEW_SPEC = importlib.util.spec_from_file_location(
    "indory_ocr_benchmark_review", ROOT / "benchmark" / "review.py"
)
assert REVIEW_SPEC is not None
review = importlib.util.module_from_spec(REVIEW_SPEC)
assert REVIEW_SPEC.loader is not None
REVIEW_SPEC.loader.exec_module(review)

REVIEW_SERVER_SPEC = importlib.util.spec_from_file_location(
    "indory_ocr_benchmark_review_server", ROOT / "benchmark" / "review_server.py"
)
assert REVIEW_SERVER_SPEC is not None
review_server = importlib.util.module_from_spec(REVIEW_SERVER_SPEC)
assert REVIEW_SERVER_SPEC.loader is not None
sys.modules[REVIEW_SERVER_SPEC.name] = review_server
REVIEW_SERVER_SPEC.loader.exec_module(review_server)


class BenchmarkRunnerTest(unittest.TestCase):
    def test_parse_modes_accepts_aliases(self) -> None:
        self.assertEqual(bench.parse_modes("ocr_llm,waybill"), ["waybill"])
        self.assertEqual(bench.parse_modes("llm"), ["waybill"])

    def test_waybill_expected_room_and_floor_are_normalized(self) -> None:
        summary = {
            "destination": "5F 528-1호",
            "destination_floor": "5F",
            "destination_room": "528-1호",
            "needs_manual_review": False,
        }
        result = bench.evaluate(
            "waybill",
            summary,
            {"destination_floor": "5", "destination_room": "528-1", "needs_manual_review": False},
        )
        self.assertTrue(result["evaluated"])
        self.assertTrue(result["pass"])

    def test_failure_analysis_reports_low_confidence(self) -> None:
        summary = {
            "destination_room": "C102호",
            "destination_floor": "1F",
            "confidence": 0.42,
            "needs_manual_review": True,
            "risk_reasons": ["low_confidence"],
        }
        expected = {"destination_room": "B104호", "destination_floor": "B1F"}
        evaluation = bench.evaluate("waybill", summary, expected)
        analysis = bench.analyze_failure(
            ok=True,
            status_code=200,
            error=None,
            summary=summary,
            evaluation=evaluation,
            expected=expected,
            response={},
            low_confidence_threshold=0.75,
        )
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis["primary_reason"], "low_confidence")
        self.assertIn("destination_mismatch", analysis["reasons"])

    def test_failure_analysis_reports_ocr_miss_when_debug_lacks_expected_room(self) -> None:
        summary = {
            "destination_room": "C102호",
            "confidence": 0.91,
            "needs_manual_review": False,
            "risk_reasons": [],
        }
        expected = {"destination_room": "B104호"}
        evaluation = bench.evaluate("waybill", summary, expected)
        response = {
            "debug": {
                "ocr": {
                    "items": [{"text": "C102"}, {"text": "LT-42"}],
                    "combined_results": [],
                }
            }
        }
        analysis = bench.analyze_failure(
            ok=True,
            status_code=200,
            error=None,
            summary=summary,
            evaluation=evaluation,
            expected=expected,
            response=response,
            low_confidence_threshold=0.75,
        )
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis["primary_reason"], "ocr_miss_expected_room")
        self.assertIs(analysis["details"]["ocr_contains_expected_room"], False)

    def test_expected_jsonl_matches_by_filename_and_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "expected.jsonl"
            labels.write_text(
                '{"image":"sample_001.jpg","destination_room":"528호"}\n',
                encoding="utf-8",
            )
            expected = bench.load_expected(labels)
            self.assertEqual(
                bench.expected_for_image(expected, Path("/any/path/sample_001.jpg"))["destination_room"],
                "528호",
            )

    def test_existing_manifest_json_becomes_images_and_expected_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "selected_manifest.json"
            image = Path(tmp) / "sample_001_640x480.jpg"
            image.write_bytes(b"fake")
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "sample_001",
                            "image_path": str(image),
                            "ground_truth": {
                                "sample_id": "sample_001",
                                "destination_floor": "5F",
                                "destination_room": "528-1호",
                                "destination_dong": None,
                            },
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            images = bench.manifest_images(manifest, labeled_only=False, ground_truth_only=True)
            expected = bench.load_expected(manifest)

            self.assertEqual(images, [image])
            self.assertEqual(
                bench.expected_for_image(expected, image)["destination_room"],
                "528-1호",
            )

    def test_relative_manifest_paths_resolve_from_manifest_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "test"
            image_dir.mkdir(parents=True)
            image = image_dir / "sample_001_640x480.jpg"
            image.write_bytes(b"fake")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "sample_001",
                                "image_path": "images/test/sample_001_640x480.jpg",
                                "file_name": "images/test/sample_001_640x480.jpg",
                                "ground_truth": {
                                    "destination_floor": "5F",
                                    "destination_room": "528호",
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            images = bench.manifest_images(manifest, labeled_only=False, ground_truth_only=True)
            expected = bench.load_expected(manifest)

            self.assertEqual(images, [image])
            self.assertEqual(bench.expected_for_image(expected, image)["destination_room"], "528호")

    def test_existing_evaluation_csv_matches_resized_image_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "evaluation_rows.csv"
            labels.write_text(
                "sample_id,gt_floor,gt_room,gt_dong\n"
                "sample_001,7F,702호,\n",
                encoding="utf-8",
            )
            expected = bench.load_expected(labels)
            record = bench.expected_for_image(expected, Path(tmp) / "sample_001_640x480.jpg")
            self.assertEqual(record["destination_floor"], "7F")
            self.assertEqual(record["destination_room"], "702호")
            self.assertEqual(
                bench.expected_for_image(expected, Path("/any/path/sample_001.png"))["destination_room"],
                "702호",
            )

    def test_current_gt_builder_filters_and_writes_review_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "resized_images"
            image_dir.mkdir()
            for name in [
                "source_gt_640x480.jpg",
                "manual_gt_640x480.jpg",
                "excluded_640x480.jpg",
                "low_640x480.jpg",
            ]:
                (image_dir / name).write_bytes(b"fake")

            source_manifest = root / "benchmark_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "source_gt",
                                "image_path": str(image_dir / "source_gt_640x480.jpg"),
                                "ground_truth": {"destination_room": "405호", "destination_floor": "4F"},
                            },
                            {
                                "sample_id": "manual_gt",
                                "image_path": str(image_dir / "manual_gt_640x480.jpg"),
                                "ground_truth": None,
                            },
                            {
                                "sample_id": "excluded",
                                "image_path": str(image_dir / "excluded_640x480.jpg"),
                                "ground_truth": None,
                            },
                            {
                                "sample_id": "low",
                                "image_path": str(image_dir / "low_640x480.jpg"),
                                "ground_truth": None,
                            },
                            {
                                "sample_id": "missing",
                                "image_path": str(image_dir / "missing_640x480.jpg"),
                                "ground_truth": {"destination_room": "999호"},
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            annotations = root / "annotations.jsonl"
            annotations.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "manual_gt",
                                "status": "verified",
                                "ground_truth": {"destination_room": "702호", "destination_floor": "7F"},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "sample_id": "excluded",
                                "status": "exclude",
                                "exclude_reason": "non_waybill_image",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "sample_id": "low",
                                "status": "low_confidence",
                                "candidate_ground_truth": {"destination_room": "108호"},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_manifest = root / "current_manifest.json"

            summary = gt.build_current_manifest(
                source_manifest=source_manifest,
                image_dir=image_dir,
                annotations_path=annotations,
                output_manifest=output_manifest,
                strict_annotations=True,
            )

            self.assertEqual(summary["manifest_sample_count"], 3)
            self.assertEqual(summary["ground_truth_count"], 2)
            self.assertEqual(summary["excluded_count"], 1)
            self.assertEqual(summary["missing_from_image_dir_count"], 1)
            manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
            self.assertEqual([sample["sample_id"] for sample in manifest["samples"]], ["source_gt", "manual_gt", "low"])
            self.assertEqual(bench.manifest_images(output_manifest, labeled_only=False, ground_truth_only=True), [
                image_dir / "source_gt_640x480.jpg",
                image_dir / "manual_gt_640x480.jpg",
            ])
            expected = bench.load_expected(output_manifest)
            self.assertEqual(
                bench.expected_for_image(expected, image_dir / "manual_gt_640x480.jpg")["destination_room"],
                "702호",
            )
            low_rows = (root / "review_low_confidence.jsonl").read_text(encoding="utf-8").strip().splitlines()
            excluded_rows = (
                root / "review_excluded_non_waybill.jsonl"
            ).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(low_rows), 1)
            self.assertEqual(len(excluded_rows), 1)

    def test_static_review_builder_writes_html_and_flat_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "resized_images"
            original_dir = root / "originals"
            out_dir = root / "review"
            image_dir.mkdir()
            original_dir.mkdir()
            image = image_dir / "sample_001_640x480.jpg"
            original = original_dir / "sample_001.png"
            image.write_bytes(b"fake")
            original.write_bytes(b"fake")
            manifest = root / "current_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "sample_001",
                                "image_path": str(image),
                                "original_image_path": str(original),
                                "carrier": "cj",
                                "condition": "front_clean",
                                "benchmark_status": "verified",
                                "benchmark_annotation_source": "manual_jsonl",
                                "ground_truth": {
                                    "destination_floor": "5F",
                                    "destination_room": "502호",
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = review.build_review(manifest_path=manifest, out_dir=out_dir, title="Review")

            self.assertEqual(summary["item_count"], 1)
            self.assertEqual(summary["ground_truth_count"], 1)
            html_text = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("sample_001", html_text)
            self.assertIn("../resized_images/sample_001_640x480.jpg", html_text)
            self.assertIn("review_corrections.jsonl", html_text)
            jsonl = (out_dir / "review_items.jsonl").read_text(encoding="utf-8").strip()
            self.assertIn('"destination_room": "502호"', jsonl)
            csv_text = (out_dir / "review_items.csv").read_text(encoding="utf-8")
            self.assertIn("destination_room", csv_text)

    def test_review_server_save_updates_annotations_manifest_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "resized_images"
            review_dir = root / "review"
            image_dir.mkdir()
            image = image_dir / "sample_001_640x480.jpg"
            image.write_bytes(b"fake")
            source_manifest = root / "benchmark_manifest.json"
            annotations = root / "annotations.jsonl"
            output_manifest = root / "current_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "sample_001",
                                "image_path": str(image),
                                "ground_truth": {"destination_room": "302호", "destination_floor": "3F"},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            annotations.write_text("", encoding="utf-8")
            config = review_server.ReviewConfig(
                run_dir=root,
                source_manifest=source_manifest,
                image_dir=image_dir,
                annotations=annotations,
                output_manifest=output_manifest,
                review_dir=review_dir,
                title="Review",
            )

            result = review_server.save_annotation(
                {
                    "sample_id": "sample_001",
                    "status": "verified",
                    "ground_truth": {"destination_room": "303호", "destination_floor": "3F"},
                },
                config=config,
            )

            self.assertTrue(result["ok"])
            self.assertIn('"destination_room": "303호"', annotations.read_text(encoding="utf-8"))
            manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["samples"][0]["ground_truth"]["destination_room"], "303호")
            self.assertIn("303호", (review_dir / "index.html").read_text(encoding="utf-8"))

    def test_local_labeled82_dataset_has_ground_truth_when_available(self) -> None:
        try:
            images, expected, _info = bench.resolve_dataset("labeled82", bench.DEFAULT_DATASET_ROOT, limit=0)
        except Exception as exc:
            self.skipTest(f"local benchmark dataset unavailable: {exc}")
        if not images:
            self.skipTest("local benchmark dataset has no images")
        missing = [path for path in images if not path.exists()]
        if missing:
            self.skipTest(f"local labeled82 manifest references deleted image files: {len(missing)}")
        self.assertEqual(len(images), 82)
        self.assertTrue(all(bench.expected_for_image(expected, path) for path in images))

    def test_local_groundtruth83_dataset_includes_qc_excluded_ground_truth_when_available(self) -> None:
        try:
            images, expected, _info = bench.resolve_dataset("groundtruth83", bench.DEFAULT_DATASET_ROOT, limit=0)
        except Exception as exc:
            self.skipTest(f"local benchmark dataset unavailable: {exc}")
        if not images:
            self.skipTest("local benchmark dataset has no images")
        missing = [path for path in images if not path.exists()]
        if missing:
            self.skipTest(f"local groundtruth83 manifest references deleted image files: {len(missing)}")
        self.assertEqual(len(images), 83)
        self.assertTrue(all(bench.expected_for_image(expected, path) for path in images))

    def test_local_current_verified_dataset_has_current_ground_truth_when_available(self) -> None:
        manifest = bench.DEFAULT_DATASET_ROOT / "run_full_640x480" / "current_manifest.json"
        if not manifest.exists():
            self.skipTest("current manifest is not built; run benchmark/gt.py build")
        images, expected, _info = bench.resolve_dataset("current_verified", bench.DEFAULT_DATASET_ROOT, limit=0)
        if not images:
            self.skipTest("local current_verified dataset has no images")
        self.assertTrue(all(path.exists() for path in images))
        self.assertTrue(all(bench.expected_for_image(expected, path) for path in images))


if __name__ == "__main__":
    unittest.main()
