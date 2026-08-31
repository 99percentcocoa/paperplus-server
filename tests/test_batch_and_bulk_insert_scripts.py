import importlib.util
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
