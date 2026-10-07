"""Parity: with the same seed, app.generation produces exactly the worksheet JSON the old repo's
generator (worksheet_json_generator.py) does -- every homework level A-G and practice level, in
English and Marathi. Skipped when the old repo or its venv isn't present (e.g. in docker).

The old code runs in a subprocess under the old repo's own venv (it needs cv2/apriltags via
models.py). Its .env points DATABASE_URL at production; the generator never touches a DB, but
DATABASE_URL is overridden with an unusable value anyway and the reference script fails if any
DB driver module was imported.
"""

import json
import os
import random
import subprocess
from pathlib import Path

import pytest

from app.generation.composition import HOMEWORK_LEVELS, PRACTICE_THEME_SKILLS, compose_worksheet

OLD_REPO = Path(__file__).resolve().parents[3]
OLD_PYTHON = OLD_REPO / ".venv" / "bin" / "python"
REFERENCE_SCRIPT = Path(__file__).with_name("_old_generator_reference.py")

pytestmark = pytest.mark.skipif(
    not (OLD_PYTHON.exists() and (OLD_REPO / "worksheet_json_generator.py").exists()),
    reason="old repo / its venv not available",
)

HOMEWORK_SEEDS = range(15)
PRACTICE_SEEDS = range(6)


def _cases() -> list[tuple[str, str, str, int]]:
    cases = []
    for language in ("en", "mr"):
        for level in HOMEWORK_LEVELS:
            cases += [("homework", level, language, seed) for seed in HOMEWORK_SEEDS]
        for theme, levels in PRACTICE_THEME_SKILLS.items():
            for level in levels:
                cases += [("practice", f"{theme}{level}", language, seed) for seed in PRACTICE_SEEDS]
    return cases


@pytest.fixture(scope="module")
def old_output(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("parity")
    cases_path, out_path = work_dir / "cases.json", work_dir / "old.json"
    cases_path.write_text(json.dumps(_cases()), encoding="utf-8")
    env = {**os.environ, "DATABASE_URL": "postgresql://parity-test-must-not-connect.invalid/none"}
    result = subprocess.run(
        [str(OLD_PYTHON), str(REFERENCE_SCRIPT), str(cases_path), str(out_path)],
        cwd=OLD_REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"old generator failed:\n{result.stderr[-3000:]}"
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_generator_matches_old_code_for_every_level_language_and_seed(old_output):
    cases = _cases()
    assert len(old_output) == len(cases)
    mismatches = []
    for (worksheet_type, level, language, seed), expected in zip(cases, old_output):
        actual = compose_worksheet(worksheet_type, level, language, random.Random(seed))
        if actual != expected:
            mismatches.append((worksheet_type, level, language, seed))
    assert not mismatches, f"{len(mismatches)}/{len(cases)} worksheets differ, e.g. {mismatches[:5]}"
