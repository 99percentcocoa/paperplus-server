"""Run by tests/test_generation_parity.py under the OLD repo's venv (cwd = old repo root), never
imported by pytest. Generates worksheets with the old worksheet_json_generator for each requested
(type, level, language, seed) case listed in the JSON file argv[1] and writes them as JSON to argv[2].

The old code reads randomness from the global `random` module; seeding it gives the same draw
sequence as app.generation's `random.Random(seed)`. The old code runs completely unmodified.
"""

import json
import random
import sys

sys.path.insert(0, ".")

from worksheet_json_generator import create_practice_worksheet_json, create_worksheet_json  # noqa: E402

with open(sys.argv[1], encoding="utf-8") as f:
    cases = json.load(f)
out = []
real_stdout = sys.stdout
sys.stdout = sys.stderr  # the old code print()s diagnostics
for worksheet_type, level, language, seed in cases:
    random.seed(seed)
    if worksheet_type == "homework":
        sheet = create_worksheet_json(f"Worksheet Level {level}", level, language, worksheet_category="homework")
    else:
        sheet = create_practice_worksheet_json(f"Practice Worksheet {level}", level[0], int(level[1:]), language)
    out.append(sheet)
sys.stdout = real_stdout

leaked = sorted(m for m in sys.modules if m.split(".")[0] in {"sqlalchemy", "psycopg", "psycopg2", "db"})
if leaked:
    raise SystemExit(f"old generator imported DB modules: {leaked}")

with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
