import unittest
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waybill_ocr_llm.candidates import (
    CODE_LIKE_RE,
    context_room_texts,
    destination_candidates,
    has_address_context,
    parse_candidate_floor,
    parse_candidate_dong,
    parse_candidate_dongs,
    room_floor,
    room_allowed_by_policy,
    room_candidates_from_digits,
    room_candidates_from_hyphen_token,
)
from waybill_ocr_llm.llm import (
    ADDRESS_HINT_RE,
    CODE_RE,
    decision_from_candidate_response,
    infer_floor_from_room,
    layout_decision_from_candidates,
    validate_llm_response,
)
from waybill_ocr_llm.ocr import require_paddleocr


class WaybillCandidateAddressContextTest(unittest.TestCase):
    def test_test_fixture_words_alone_are_not_address_context(self) -> None:
        self.assertFalse(has_address_context("로봇대로"))
        self.assertFalse(has_address_context("테스트로"))
        self.assertFalse(has_address_context("한빛 70501"))
        self.assertEqual(context_room_texts("한빛 70501"), [])

    def test_address_structure_recovers_concatenated_room_digits(self) -> None:
        self.assertTrue(has_address_context("로봇대로 70501"))
        self.assertEqual(context_room_texts("로봇대로 70501"), ["501호"])
        self.assertEqual(context_room_texts("배송주소 로봇대로 70501"), ["501호"])

    def test_llm_address_hint_regex_is_structural(self) -> None:
        self.assertIsNone(ADDRESS_HINT_RE.search("로봇대로"))
        self.assertIsNone(ADDRESS_HINT_RE.search("서울"))
        self.assertIsNotNone(ADDRESS_HINT_RE.search("로봇대로 70501"))
        self.assertIsNotNone(ADDRESS_HINT_RE.search("서울특별시 강남구"))

    def test_llm_tracking_code_regex_is_not_prefix_whitelist_only(self) -> None:
        self.assertIsNotNone(CODE_RE.search("Y247"))
        self.assertIsNotNone(CODE_RE.search("LZ-3509"))
        self.assertIsNotNone(CODE_RE.search("AB-1234"))
        self.assertIsNone(CODE_RE.search("528호"))

    def test_candidate_tracking_code_regex_is_not_prefix_whitelist_only(self) -> None:
        self.assertIsNotNone(CODE_LIKE_RE.search("Y247"))
        self.assertIsNotNone(CODE_LIKE_RE.search("LZ-3509"))
        self.assertIsNotNone(CODE_LIKE_RE.search("AB-1234"))
        self.assertIsNone(CODE_LIKE_RE.search("528호"))

    def test_paddleocr_is_required_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(require_paddleocr())
        with patch.dict("os.environ", {"WAYBILL_OCR_REQUIRE_PADDLE": "0"}, clear=True):
            self.assertFalse(require_paddleocr())

    def test_non_destination_numbers_are_not_room_fallbacks(self) -> None:
        self.assertFalse(has_address_context("운송장번호 70501"))
        self.assertEqual(context_room_texts("운송장번호 70501"), [])
        self.assertEqual(context_room_texts("528호"), ["528호"])

    def test_room_policy_is_not_fifth_floor_only_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(room_allowed_by_policy("1205호"))
            self.assertTrue(room_allowed_by_policy("528-1호"))
        with patch.dict("os.environ", {"WAYBILL_ROOM_POLICY": "5xx"}, clear=True):
            self.assertFalse(room_allowed_by_policy("1205호"))
            self.assertTrue(room_allowed_by_policy("528-1호"))

    def test_hyphen_recovery_is_generic_and_prefix_recovery_is_opt_in(self) -> None:
        self.assertIn("430-1호", room_candidates_from_digits("4301"))
        self.assertIn("528-1호", room_candidates_from_digits("5281"))
        with patch.dict("os.environ", {"WAYBILL_MAX_REASONABLE_FLOOR": "60"}, clear=True):
            self.assertEqual(room_candidates_from_digits("4301"), ["4301호"])
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(room_candidates_from_hyphen_token("T28-1"), [])
        with patch.dict("os.environ", {"WAYBILL_ROOM_PREFIX_FLOOR_DIGIT": "5"}, clear=True):
            self.assertEqual(room_candidates_from_hyphen_token("T28-1"), ["528-1호"])

    def test_dong_is_not_shortened_from_long_number(self) -> None:
        self.assertEqual(parse_candidate_dong("8101동"), "8101동")
        self.assertEqual(parse_candidate_dong("배송주소 101동"), "101동")

    def test_glued_road_number_dong_adds_tail_candidate_without_hardcoded_address(self) -> None:
        self.assertEqual(parse_candidate_dongs("서울시 중구 샘플길 6101동 702호")[:2], ["101동", "6101동"])
        self.assertEqual(parse_candidate_dongs("A동 702호"), ["A동"])
        self.assertEqual(parse_candidate_dongs("A동702호"), ["A동"])
        self.assertEqual(parse_candidate_dongs("C통 911호"), ["C동"])
        self.assertEqual(parse_candidate_dongs("받는분 주소 A8 702호 부산광역시 중구"), ["A동"])
        self.assertEqual(parse_candidate_dongs("받는분 주소 14 통 702호 부산광역시 중구")[0], "A동")

    def test_basement_floor_is_preserved_for_room_candidates(self) -> None:
        self.assertIsNone(room_floor("B201호"))
        self.assertIsNone(infer_floor_from_room("B201호"))
        self.assertEqual(parse_candidate_floor("주소 지하 2중 B201호")[0], "B2F")
        self.assertEqual(parse_candidate_floor("주소 3중 B201호")[0], "3F")
        candidates = destination_candidates(
            [
                {"text": "주소 지하 2층 8201 서울특별시 동작구 성대로 65", "confidence": 0.95},
            ]
        )
        self.assertTrue(
            any(
                candidate.get("destination_room") == "B201호"
                and candidate.get("destination_floor") == "B2F"
                for candidate in candidates
            )
        )
        ambiguous_candidates = destination_candidates(
            [
                {"text": "주소 B201호 서울특별시 동작구 성대로 65", "confidence": 0.95},
            ]
        )
        self.assertTrue(
            any(
                candidate.get("destination_room") == "B201호"
                and candidate.get("destination_floor") is None
                for candidate in ambiguous_candidates
            )
        )
        self.assertTrue(
            any(
                candidate.get("destination_room") == "B201호"
                and candidate.get("destination_floor") == "B2F"
                for candidate in ambiguous_candidates
            )
        )
        candidates_without_floor_glyph = destination_candidates(
            [
                {"text": "주소 지하 2 B201 서울특별시 동작구 성대로 65", "confidence": 0.95},
            ]
        )
        self.assertTrue(
            any(
                candidate.get("destination_room") == "B201호"
                and candidate.get("destination_floor") == "B2F"
                for candidate in candidates_without_floor_glyph
            )
        )

    def test_date_like_numbers_are_not_room_candidates(self) -> None:
        self.assertEqual(context_room_texts("접수일 2024.07.04 배송주소 서울특별시 중구"), [])
        self.assertIn("1201호", room_candidates_from_digits("12018"))

    def test_ocr_decimal_noise_can_still_recover_room_digits(self) -> None:
        self.assertIn("405호", context_room_texts("받는분 주소 서울특별시 중구 405.0"))
        self.assertIn("405호", context_room_texts("받는분 주소 서울특별시 중구 405."))
        self.assertIn("1503호", context_room_texts("받는분 주소 서울특별시 중구 1503.호"))
        self.assertIn("404호", context_room_texts("받는분 주소 서울특별시 중구 2.404 00-"))
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 서울특별시 중구 샘플로 19 405.0", "confidence": 0.95},
                {"text": "주문번호 ORD-2026-0405", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "405호")
        self.assertIn("108호", context_room_texts("받는분 주소 대전광역시 유성구 성로 37 108R"))

    def test_attached_dong_room_can_add_short_room_candidate(self) -> None:
        self.assertIn("301호", context_room_texts("받는분 주소 서울시 중구 101동3012"))
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 광주광역시 상무로 33 101동 1201", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "1201호")
        self.assertEqual(candidates[0]["destination_dong"], "101동")

    def test_variant_context_does_not_borrow_floor_from_other_preprocess_image(self) -> None:
        candidates = destination_candidates(
            [
                {
                    "text": "받는분 주소 서울특별시 중구 샘플로 12 B201호",
                    "confidence": 0.95,
                    "variant_image": "original.png",
                },
                {
                    "text": "주소 3층",
                    "confidence": 0.95,
                    "variant_image": "full_x2.png",
                },
            ]
        )
        self.assertTrue(any(candidate.get("destination_room") == "B201호" for candidate in candidates))
        self.assertFalse(
            any(
                candidate.get("destination_room") == "B201호"
                and candidate.get("destination_floor") == "3F"
                for candidate in candidates
            )
        )

    def test_h_prefixed_ocr_room_is_recovered_but_not_inside_sorting_code(self) -> None:
        self.assertEqual(context_room_texts("받는분 주소 서울특별시 중구 H631"), ["631호"])
        self.assertEqual(context_room_texts("받는분 주소 서울특별시 중구 HJ-41 H631"), [])
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 서울특별시 중구 H631", "confidence": 0.95},
                {"text": "분류코드 HJ-41 H631", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "631호")

    def test_address_fallback_ignores_year_like_twenty_prefix_numbers(self) -> None:
        self.assertEqual(context_room_texts("받는분 주소 서울특별시 중구 샘플로 12 20130"), [])
        self.assertIn("1201호", context_room_texts("받는분 주소 서울특별시 중구 샘플로 12 12018"))

    def test_leading_zero_room_prefers_full_plausible_room_over_tail(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 서울특별시 중구 샘플로 2 001321호", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "1321호")
        self.assertNotEqual(candidates[0]["destination_room"], "321호")

    def test_suffix_zero_room_recovery_does_not_beat_explicit_plausible_room(self) -> None:
        self.assertIn("514호", context_room_texts("받는분 주소 광주광역시 북구 생품로 64 5140"))
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 서울특별시 중구 샘플로 10 1010호", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "1010호")
        self.assertIn("101호", [candidate.get("destination_room") for candidate in candidates])

    def test_ab_r_ocr_recovers_a_dong_room_without_treating_separate_u_as_dong(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 부산광역시 중구 AB 702R", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_dong"], "A동")
        self.assertEqual(candidates[0]["destination_room"], "702호")
        self.assertEqual(parse_candidate_dongs("받는분 주소 부산광역시 중구 U B104R"), [])
        b_candidates = destination_candidates(
            [
                {"text": "받는분 주소 부산광역시 중구 U B104R", "confidence": 0.95},
            ]
        )
        self.assertTrue(all(candidate.get("destination_dong") != "U동" for candidate in b_candidates))
        self.assertEqual(b_candidates[0]["destination_room"], "B104호")

        result = {
            "ocr_items": [
                {"text": "C102 u7010m U B104R w0 수업", "confidence": 0.95},
                {"text": "MARNI ET-200X-4821 145-20", "confidence": 0.95},
            ]
        }
        layout_candidates = destination_candidates(result["ocr_items"])
        decision = layout_decision_from_candidates(result, layout_candidates)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["destination_room"], "B104호")

    def test_noisy_latin_dong_room_separator_recovers_room(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "CR.911量", "confidence": 0.95},
                {"text": "B-22 HJ68", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_dong"], "C동")
        self.assertEqual(candidates[0]["destination_room"], "911호")
        self.assertEqual(candidates[0]["destination_floor"], "9F")
        self.assertEqual(destination_candidates([{"text": "C102 A317", "confidence": 0.95}]), [])

    def test_sorting_suffix_and_artificial_hyphen_do_not_beat_real_room_shape(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "분류코드 B21-111 8823 배송주소 서울특별시 중구 샘플로 12", "confidence": 0.95},
                {"text": "405호", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "405호")

        noisy_candidates = destination_candidates(
            [
                {"text": "받는분 주소 서울특별시 중구 샘플로 7 7303.호", "confidence": 0.95},
            ]
        )
        self.assertEqual(noisy_candidates[0]["destination_room"], "303호")
        self.assertLess(
            next(
                candidate["score"]
                for candidate in noisy_candidates
                if candidate.get("destination_room") == "730-3호"
            ),
            noisy_candidates[0]["score"],
        )

    def test_receiver_numeric_name_does_not_beat_floor_only_destination(self) -> None:
        result = {
            "ocr_items": [
                {"text": "받는분 221", "confidence": 0.95},
                {"text": "서울특별시 강남구 테헤란로", "confidence": 0.95},
                {"text": "3층", "confidence": 0.95},
            ]
        }
        candidates = destination_candidates(result["ocr_items"])
        decision = layout_decision_from_candidates(result, candidates)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["destination_floor"], "3F")
        self.assertIsNone(decision.get("destination_room"))

    def test_ocr_eight_can_recover_basement_room_without_forcing_floor(self) -> None:
        self.assertIn("B104호", room_candidates_from_digits("8104"))
        self.assertIn("B201호", room_candidates_from_digits("8201"))
        self.assertNotIn("B514호", room_candidates_from_digits("8514"))

    def test_short_dong_does_not_match_inside_longer_dong(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "주소 부산광역시 해운대구 테스트길 8 01동 514호", "confidence": 0.95},
                {"text": "주소 부산광역시 해운대구 테스트길 8 101동 514호", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_dong"], "101동")

    def test_glued_road_number_dong_prefers_tail_dong(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 서울특별시 은하구 연회로 6101동702호", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "702호")
        self.assertEqual(candidates[0]["destination_dong"], "101동")

    def test_glued_road_number_room_prefers_tail_room(self) -> None:
        candidates = destination_candidates(
            [
                {"text": "받는분 주소 전주시 덕진구 기대로 58403호", "confidence": 0.95},
                {"text": "주문번호 3155-0 접수일 2026-05-25", "confidence": 0.95},
            ]
        )
        self.assertEqual(candidates[0]["destination_room"], "403호")

    def test_direct_room_recovers_from_conflicting_floor_only_candidate_id(self) -> None:
        candidates = [
            {
                "candidate_id": 0,
                "destination_room": "302호",
                "destination_floor": "3F",
                "destination_dong": None,
                "floor_source": "room_inferred",
                "evidence_indices": [0],
                "address_text": "광주광역시 빛가람구 상무 42302",
                "score": 4.0,
            },
            {
                "candidate_id": 1,
                "destination_room": None,
                "destination_floor": "1F",
                "destination_dong": None,
                "floor_source": "explicit",
                "evidence_indices": [9],
                "address_text": "전라남도 시산단중앙로7 집하장 1층",
                "score": 0.7,
            },
        ]
        response = {
            "candidate_id": "1",
            "destination_room": "302호",
            "destination_floor": None,
            "destination_dong": None,
            "confidence": 0.9,
        }
        decision = decision_from_candidate_response(response, candidates)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["destination_room"], "302호")
        self.assertEqual(decision["destination_floor"], "3F")
        self.assertEqual(validate_llm_response(response, candidates, decision), [])


if __name__ == "__main__":
    unittest.main()
