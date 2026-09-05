import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

from template_layouts import get_handwritten_field_roi, get_template_layout

ROOT = Path(__file__).resolve().parents[1]


def load_worksheet_pdf_generator_module():
    return load_module("worksheet_pdf_generator", ROOT / "worksheet_pdf_generator.py")


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchAndBulkInsertScriptTests(unittest.TestCase):
    def test_batch_generate_level_spec_handles_practice_and_homework(self):
        batch = load_module("batch_generate_worksheets", ROOT / "batch_generate_worksheets.py")

        self.assertEqual(
            batch.resolve_level_spec("A", "homework"),
            {"level": "A", "worksheet_category": "homework"},
        )
        self.assertEqual(
            batch.resolve_level_spec("D5", "practice"),
            {"theme": "D", "level": 5, "worksheet_category": "practice"},
        )

    def test_batch_generate_supports_multiple_sheets_per_level(self):
        batch = load_module("batch_generate_worksheets", ROOT / "batch_generate_worksheets.py")

        self.assertEqual(
            batch.build_output_filename("en", "homework", "A", worksheet_id=8001, index=0),
            "8001_en.json",
        )
        self.assertEqual(
            batch.build_output_filename("en", "homework", "A", worksheet_id=None, index=3),
            "en_homework_A_3.json",
        )

    def test_bulk_insert_detects_generated_filenames(self):
        bulk = load_module("bulk_insert_worksheets", ROOT / "bulk_insert_worksheets.py")

        self.assertEqual(
            bulk.parse_generated_filename("8001_en.json"),
            {"worksheet_id": 8001, "language": "en", "worksheet_category": None},
        )
        self.assertEqual(
            bulk.parse_generated_filename("8002_mr.json"),
            {"worksheet_id": 8002, "language": "mr", "worksheet_category": None},
        )
        self.assertEqual(
            bulk.parse_generated_filename("en_practice_D5.json"),
            {"worksheet_id": None, "language": "en", "worksheet_category": "practice"},
        )
        self.assertEqual(
            bulk.parse_generated_filename("mr_homework_A.json"),
            {"worksheet_id": None, "language": "mr", "worksheet_category": "homework"},
        )
        self.assertEqual(
            bulk.parse_generated_filename("en_homework_A_3.json"),
            {"worksheet_id": None, "language": "en", "worksheet_category": "homework"},
        )

    def test_bulk_insert_reads_from_subfolder(self):
        bulk = load_module("bulk_insert_worksheets", ROOT / "bulk_insert_worksheets.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_dir = root / "files" / "json" / "nested"
            json_dir.mkdir(parents=True)
            target = json_dir / "8001_en.json"
            target.write_text('{"worksheet_id": 8001, "worksheet_category": "homework"}', encoding="utf-8")
            (json_dir / "ignore.txt").write_text("skip", encoding="utf-8")

            bulk.JSON_DIR = root / "files" / "json"

            files = bulk.iter_generated_files(subfolder="nested")
            self.assertEqual([p.name for p in files], ["8001_en.json"])
            self.assertEqual(files[0].parent.name, "nested")

    def test_resolve_template_filename_allows_basic_omr_without_language(self):
        generator = load_worksheet_pdf_generator_module()

        self.assertEqual(
            generator._resolve_template_filename({"worksheet_category": "omr"}, "basic_omr", None),
            ("basic_omr", "template_en.html"),
        )

    def test_basic_omr_prefers_actual_question_list_length_over_stale_question_count(self):
        generator = load_worksheet_pdf_generator_module()
        worksheet = {
            "worksheet_category": "omr",
            "template_name": "basic_omr",
            "question_count": 20,
            "questions": [{"index": i, "question_text": "", "options": ["", "", "", ""], "correct_option": ""} for i in range(1, 31)],
        }

        self.assertEqual(generator._resolve_question_count(worksheet), 30)

    def test_basic_omr_handles_more_than_available_row_tags(self):
        generator = load_worksheet_pdf_generator_module()

        html = generator.generate_basic_omr_questions_html(worksheet_id=1, question_count=39)

        self.assertIn("tag25_09_", html)
        self.assertGreaterEqual(html.count("class='row-marker'"), 13)

    def test_template_layouts_override_roi_settings_per_template(self):
        layout = get_template_layout("basic_omr")
        self.assertGreaterEqual(len(layout.question_roi_columns), 3)
        self.assertEqual(layout.question_roi_columns[0], (50, -40, 365, 90))
        self.assertEqual(layout.question_roi_columns[1], (430, -40, 365, 90))
        self.assertEqual(layout.question_roi_columns[2], (810, -40, 365, 90))
        self.assertTrue(get_handwritten_field_roi("basic_omr", "question_paper_code") is None or isinstance(get_handwritten_field_roi("basic_omr", "question_paper_code"), tuple))

    def test_row_tag_decoder_supports_legacy_and_13_tag_layout(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")

        legacy_row_tags = image_service.worksheet_id_to_rows(1234)
        self.assertEqual(image_service.decode_row_tags(legacy_row_tags), 1234)

        extended_row_tags = legacy_row_tags + [2, 5, 1]
        decoded = image_service.decode_row_tag_metadata(extended_row_tags)
        self.assertEqual(decoded["worksheet_id"], 1234)
        self.assertEqual(decoded["page_no"], 2)
        self.assertEqual(decoded["first_question_index"], 40)
        self.assertTrue(all(v <= 34 for v in extended_row_tags))

        with self.assertRaises(ValueError):
            image_service.decode_row_tags([1, 2, 3])

    def test_basic_omr_pages_continue_question_numbers(self):
        generator = load_worksheet_pdf_generator_module()

        html = generator.generate_basic_omr_questions_html(
            worksheet_id=1,
            question_count=39,
            page_no=2,
            first_question_index=40,
        )

        self.assertIn("40.", html)
        self.assertGreaterEqual(html.count("class='question_td'"), 3)

    def test_omr_row_tags_include_page_metadata_for_page_two(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")

        row_tags = image_service.worksheet_id_to_rows(1234, page_no=2, first_question_index=40)
        self.assertEqual(len(row_tags), 13)
        self.assertTrue(all(v <= 34 for v in row_tags))
        self.assertEqual(image_service.decode_row_tag_metadata(row_tags)["page_no"], 2)
        self.assertEqual(image_service.decode_row_tag_metadata(row_tags)["first_question_index"], 40)

    def test_page_one_metadata_does_not_allow_question_140_start_index(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")

        row_tags = [0, 0, 4, 2, 30, 24, 14, 12, 24, 0, 0, 0, 4]
        decoded = image_service.decode_row_tag_metadata(row_tags)

        self.assertEqual(decoded["page_no"], 1)
        self.assertEqual(decoded["first_question_index"], 1)

    def test_question_paper_code_validation_requires_single_uppercase_letter_a_to_f(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")

        self.assertEqual(image_service.validate_question_paper_code("A"), "A")
        self.assertEqual(image_service.validate_question_paper_code(" a "), "A")
        self.assertEqual(image_service.validate_question_paper_code("F"), "F")
        self.assertEqual(image_service.validate_question_paper_code("AB"), "")
        self.assertEqual(image_service.validate_question_paper_code("g"), "")
        self.assertEqual(image_service.validate_question_paper_code("3"), "")

    def test_omr_v2_row_metadata_uses_basic_omr_template_when_worksheet_not_in_db(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")

        row_metadata = {"worksheet_id": 9999, "page_no": 2, "first_question_index": 40, "format": "omr_v2"}

        self.assertEqual(image_service.infer_template_name_from_scan(9999, row_metadata, None), "basic_omr")

    def test_setd_page1_roll_number_ocr_detects_9876(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")
        input_path = ROOT / "testing" / "images" / "setD_page1.jpeg"

        worksheet = image_service.scan_image(image_service.InputImageMeta(image_path=str(input_path)))

        self.assertEqual(worksheet.template_name, "basic_omr")
        self.assertEqual(worksheet.roll_number, "9876")

    def test_setd_page1_question_paper_code_ocr_detects_d(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")
        input_path = ROOT / "testing" / "images" / "setD_page1.jpeg"

        worksheet = image_service.scan_image(image_service.InputImageMeta(image_path=str(input_path)))

        self.assertEqual(worksheet.template_name, "basic_omr")
        self.assertEqual(worksheet.question_paper_code, "D")

    def test_basic_omr_uses_question_paper_code_to_resolve_answer_key(self):
        from db.worksheets import save_omr_answer_key, resolve_answer_key_for_template

        worksheet_id = 9003
        save_omr_answer_key("basic_omr", "A", ["A", "B", "C", "D"], worksheet_id=worksheet_id)
        save_omr_answer_key("basic_omr", "B", ["D", "C", "B", "A"], worksheet_id=worksheet_id)

        self.assertEqual(resolve_answer_key_for_template("basic_omr", worksheet_id, "A"), ["A", "B", "C", "D"])
        self.assertEqual(resolve_answer_key_for_template("basic_omr", worksheet_id, "B"), ["D", "C", "B", "A"])
        self.assertIsNone(resolve_answer_key_for_template("basic_omr", worksheet_id, "Z"))

    def test_get_combined_worksheet_answers_merges_page_1_and_page_2_for_worksheet_4810(self):
        from db.connection import get_connection
        from db.submissions import get_combined_worksheet_answers

        page1 = [{"question_index": i, "selected_option": "A", "is_correct": True} for i in range(1, 40)]
        page2 = [{"question_index": i, "selected_option": "B", "is_correct": False} for i in range(40, 79)]
        inserted = []
        student_ids = ["0001", "0002"]

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for student_id, answers in zip(student_ids, [page1, page2]):
                        cur.execute(
                            """
                            INSERT INTO submissions (student_id, worksheet_id, score, from_number, answers_json, worksheet_category)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING submission_id
                            """,
                            (student_id, 4810, len(answers), "", json.dumps(answers), "omr"),
                        )
                        inserted.append(cur.fetchone()["submission_id"])

            combined = get_combined_worksheet_answers(4810)
            indices = [item["question_index"] for item in combined]

            self.assertEqual(indices[:3], [1, 2, 3])
            self.assertEqual(indices[-3:], [76, 77, 78])
            self.assertEqual(len(combined), 78)
            self.assertEqual(min(indices), 1)
            self.assertEqual(max(indices), 78)
        finally:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if inserted:
                        cur.execute(
                            "DELETE FROM submissions WHERE submission_id = ANY(%s)",
                            (inserted,),
                        )

    def test_build_submission_answers_keeps_page_two_question_numbers(self):
        from offline_pipeline import build_submission_answers

        payload = build_submission_answers(["A", "", "C"], first_question_index=40)

        self.assertEqual([item["question_index"] for item in payload], [40, 41, 42])
        self.assertEqual(payload[0]["selected_option"], "A")
        self.assertEqual(payload[1]["selected_option"], "")
        self.assertEqual(payload[2]["selected_option"], "C")

    def test_basic_omr_grading_uses_first_question_index_and_truncates_short_answer_key(self):
        from services.grading_service import get_answer_key_for_question_slice

        full_answer_key = [chr(ord("A") + (i % 4)) for i in range(78)]

        page_one_slice = get_answer_key_for_question_slice(full_answer_key, 1, 39)
        self.assertEqual(page_one_slice, ["A", "B", "C", "D"] * 9 + ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"])

        page_two_slice = get_answer_key_for_question_slice(full_answer_key, 40, 39)
        self.assertEqual(page_two_slice[0], "A")
        self.assertEqual(page_two_slice[-1], "C")
        self.assertEqual(len(page_two_slice), 39)

        short_key = ["A", "B", "C"]
        self.assertEqual(get_answer_key_for_question_slice(short_key, 1, 10), ["A", "B", "C"])

    def test_insert_questions_for_worksheet_allows_omr_questions_without_skill_code(self):
        from db.flows import add_worksheet_to_db
        from db.questions import get_questions_for_worksheet

        worksheet = {
            "title": "OMR Regression",
            "worksheet_category": "omr",
            "template_name": "basic_omr",
            "questions": [
                {"index": 1, "question_text": "", "options": ["", "", "", ""], "correct_option": "A"},
                {"index": 2, "question_text": "", "options": ["", "", "", ""], "correct_option": "B"},
            ],
        }

        worksheet_id = 990000 + (time.time_ns() % 100000)
        result = add_worksheet_to_db(worksheet, worksheet_id=worksheet_id, worksheet_category="omr")
        self.assertEqual(len(result["question_ids"]), 2)

        rows = get_questions_for_worksheet(result["worksheet_id"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["skill_code"] == "omr" for r in rows))


if __name__ == "__main__":
    unittest.main()
