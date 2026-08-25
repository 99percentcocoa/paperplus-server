import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StudentImportHelpersTests(unittest.TestCase):
    def test_student_ids_are_zero_padded_and_start_at_0002(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "import_students_from_csv",
            ROOT / "import_students_from_csv.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.normalize_student_id("2"), "0002")
        self.assertEqual(module.normalize_student_id(7), "0007")
        self.assertEqual(module.next_student_id(["0002", "0009"]), "0010")
        self.assertEqual(module.next_student_id([]), "0002")

    def test_student_name_extraction_ignores_blank_rows(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "import_students_from_csv",
            ROOT / "import_students_from_csv.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        row = {"Name\nनाव": "Archana Mithun Lhange", "STD.\nइ.": "2"}
        self.assertEqual(module.extract_student_name(row), "Archana Mithun Lhange")
        self.assertEqual(module.extract_student_name({"Name\nनाव": "", "STD.\nइ.": "2"}), None)
        self.assertEqual(module.extract_student_name({"STD.\nइ.": "2"}), None)


if __name__ == "__main__":
    unittest.main()
