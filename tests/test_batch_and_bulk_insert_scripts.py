import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


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

    def test_config_uses_repo_relative_paths_without_system_specific_refs(self):
        import config

        root = Path(config.__file__).resolve().parent
        with mock.patch.dict(os.environ, {"PAPERPLUS_PROJECT_ROOT": "/tmp/colab_project"}, clear=False):
            self.assertEqual(config.resolve_project_path("assets", "tags"), "/tmp/colab_project/assets/tags")
            self.assertEqual(config.resolve_project_path("files", "json"), "/tmp/colab_project/files/json")

        with mock.patch.dict(
            os.environ,
            {
                "TAGS_PATH": str(root / "assets" / "tags"),
                "PAPERPLUS_PROJECT_ROOT": "/tmp/colab_project",
            },
            clear=False,
        ):
            self.assertEqual(config.resolve_config_path("TAGS_PATH", "assets", "tags"), "/tmp/colab_project/assets/tags")

        with mock.patch.dict(
            os.environ,
            {
                "TAGS_PATH": "/home/saarang/paperplus_server/assets/tags",
                "PAPERPLUS_PROJECT_ROOT": "/tmp/colab_project",
            },
            clear=False,
        ):
            self.assertEqual(config.resolve_config_path("TAGS_PATH", "assets", "tags"), "/tmp/colab_project/assets/tags")
            self.assertNotIn("/home/saarang", config.resolve_config_path("TAGS_PATH", "assets", "tags"))


if __name__ == "__main__":
    unittest.main()
