"""Run by tests/test_generation_render.py under the OLD repo's venv (cwd = old repo root), never
imported by pytest. Renders each case in the JSON file argv[1] with the old
worksheet_pdf_generator.generate_worksheet_pdf, writing <out_dir>/<name>.pdf (argv[2]).

The old renderer lazily imports services.image_service just for worksheet_id_to_rows, and that
module imports db.worksheets (DB access) and services.inference (OCR models). A stub module
holding only the old encoder functions -- extracted from the old source with `ast` -- is put in
its place first, so neither is ever imported.
"""

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, ".")

ENCODER_FUNCTIONS = {"encode_worksheet_id_rows", "checksum", "worksheet_id_to_rows"}
tree = ast.parse(Path("services/image_service.py").read_text(encoding="utf-8"))
stub = types.ModuleType("services.image_service")
stub.hashlib = hashlib
body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ENCODER_FUNCTIONS]
exec(compile(ast.Module(body=body, type_ignores=[]), "services/image_service.py", "exec"), stub.__dict__)
sys.modules["services.image_service"] = stub

from worksheet_pdf_generator import generate_worksheet_pdf  # noqa: E402

with open(sys.argv[1], encoding="utf-8") as f:
    cases = json.load(f)
out_dir = Path(sys.argv[2])
for case in cases:
    generate_worksheet_pdf(
        worksheet_id=case["worksheet_id"],
        worksheet_json_filename=case["json_path"],
        output_path=str(out_dir / f"{case['name']}.pdf"),
        template_name=case["template_name"],
        page_no=case.get("page_no"),
        first_question_index=case.get("first_question_index"),
    )

leaked = sorted(m for m in sys.modules if m.split(".")[0] in {"sqlalchemy", "psycopg", "psycopg2", "db"})
if leaked:
    raise SystemExit(f"old renderer imported DB modules: {leaked}")
