import importlib.util
import tempfile
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
        self.assertEqual(layout.question_roi_columns[0], (85, -40, 485, 90))
        self.assertEqual(layout.question_roi_columns[1], (355, -40, 485, 90))
        self.assertEqual(layout.question_roi_columns[2], (620, -40, 485, 90))
        self.assertTrue(get_handwritten_field_roi("basic_omr", "question_paper_code") is None or isinstance(get_handwritten_field_roi("basic_omr", "question_paper_code"), tuple))

    def test_row_tag_decoder_supports_legacy_and_13_tag_layout(self):
        image_service = load_module("services.image_service", ROOT / "services" / "image_service.py")

        legacy_row_tags = image_service.worksheet_id_to_rows(1234)
        self.assertEqual(image_service.decode_row_tags(legacy_row_tags), 1234)

        extended_row_tags = legacy_row_tags + [2, 40, 0]
        decoded = image_service.decode_row_tag_metadata(extended_row_tags)
        self.assertEqual(decoded["worksheet_id"], 1234)
        self.assertEqual(decoded["page_no"], 2)
        self.assertEqual(decoded["first_question_index"], 40)

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


if __name__ == "__main__":
    unittest.main()
