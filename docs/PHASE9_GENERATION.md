# Phase 9 — Worksheet JSON + PDF generation in v2

Status: **implemented (2026-10-08)**; steps 9.1–9.6 and 9.8 are done and verified. In 9.7 the
Dockerfile is changed but the image has **not been built** (no Docker in the dev shell). See
"Outcome" at the end.

This phase reverses the earlier "generation stays in the old repo" decision (plan item 4.2).
The goal is for v2 to make printable worksheets itself, so the old repo is no longer needed
for anything.

## What exists today (old repo)

| File | Lines | Role |
|---|---|---|
| `batch_generate_worksheets.py` | 218 | CLI: `--type homework|practice|omr --level --language --start-id --count`; writes JSON to `files/json/`, then PDF to `files/pdf/` |
| `worksheet_json_generator.py` | 545 | Level A–G → difficulty mix (`SETTINGS.WORKSHEET_LEVEL_DISTRIBUTIONS`); practice themes A/S/M/D × 1–5 (`PRACTICE_THEME_SKILLS`, with 30% carried over from the previous level); assembles 20 questions |
| `services/question_generator_service.py` | 824 | One generator per skill code (33 of them, `_gen_map`), `question_to_marathi`, `arabic_to_devanagari`, reads `services/skills.json` |
| `services/distractor_generator_service.py` | 386 | Misconception-based wrong answers per skill, with a non-negative fallback |
| `models.py: Question.choose_distractors` | ~40 | Picks 3 distractors and shuffles the correct answer's position |
| `worksheet_pdf_generator.py` | 367 | Fills the HTML template and renders it with WeasyPrint; works out the OMR page plan |
| `services/pdf_generator_service.py` | 178 | Builds the HTML for corner tags (36h11 ids 0–3), row tags (25h9) and the question grid (2 per row for regular, 3 per row for basic_omr) |
| `services/image_service.py: worksheet_id_to_rows` | — | Encodes the worksheet id and page into row tags (base-35 + checksum; extra 3 tags for page ≥ 2) |
| `assets/templates/{regular,basic_omr}/template_{en,mr}.html`, `style.css`, `assets/tags/{36h11,25h9}`, `assets/instructions/*.jpg`, `assets/logo.jpg` | — | Print assets |

### Problems in the old flow that the port should fix

1. **Worksheet ID is decided before the DB is involved.** Without `--worksheet-id`/`--start-id`,
   the PDF's row tags encode the placeholder id `0` (`PLACEHOLDER_WORKSHEET_ID`), and the DB gives the
   row a different id when the JSON is inserted later. The printed sheet then can't be matched to its
   answer key. Even with explicit ids, nothing checks that the id is still free when the JSON is
   inserted.
2. **Two-step handoff through files.** Generation writes JSON to disk, then someone runs
   `bulk_insert_worksheets.py` separately. If that second step is forgotten, or run against the
   wrong DB, sheets get printed whose answer keys don't exist (graded as `invalid_worksheet` failures).
3. **The OMR JSON doesn't match the printed pages.** OMR prints 2 pages (Q1–39, Q40–78) but
   writes a 39-question `blank_omr.json`. v2 already works around this with
   `scripts/insert_omr_worksheet.py`; generation should use the same 78-question shape.
4. **The row-tag encoder is duplicated.** It lives in the old `services/image_service.py` and
   again in `vision-service/app/vision/tags.py`, and nothing tests that the encoder and decoder agree.
5. **Rendering depends on system fonts.** `style.css` uses `font-family: "NotoSansDevanagari"`
   without an `@font-face`, so Marathi sheets only render correctly on a machine that has the
   font installed. In the `python:3.11-slim` image it would fall back to boxes ("tofu") with no error.
6. **Output isn't reproducible.** Generation uses the global `random` module with no seed, so a
   lost PDF can't be regenerated from its id.

## Target design

Generation lives in **api-service**, because it needs the DB (to reserve ids and read the
`skills` table) and owns worksheet insertion. vision-service is unchanged except for the
shared row-tag module.

```
api-service/app/generation/
  questions.py      # port of question_generator_service (_gen_map, 33 skill generators)
  distractors.py    # port of distractor_generator_service + choose_distractors
  localization.py   # Marathi numerals
  composition.py    # level/practice distributions -> list[question dict]  (pure, takes an rng)
  render.py         # HTML assembly + WeasyPrint -> PDF bytes
  service.py        # generate_worksheets(session, ...) orchestration
api-service/assets/worksheet_templates/   # templates, style.css, instructions/, logo, fonts/
api-service/assets/tags/{36h11 (ids 0-3 only), 25h9}/
shared/row_tags.py  # ONE encoder + decoder, imported by vision-service and api-service
```

**Flow: insert first, then render, then commit.**

1. Build the question JSON in memory, using a `random.Random(seed)` passed in explicitly
   (no global `random`).
2. `insert_worksheet(..., commit=False)` writes the worksheet. The DB either assigns the id or
   confirms that an explicit `--start-id` range is free.
3. Render each page's PDF using **that** id and the worksheet's real `worksheet_pages`.
4. Write the PDFs to `storage_root/worksheets/` and record the paths, the seed and the generator
   version in `worksheet_metadata`.
5. Commit only after every PDF in the batch has rendered. If any render fails, the batch is
   rolled back, so a sheet never exists in only one place.

Because the questions are stored in `worksheet_json`/`questions`, any worksheet can be
**re-rendered from the DB** later; there are no JSON files to keep around.

### CLI (replaces the three old scripts)

```bash
python scripts/generate_worksheets.py --type homework --level A --language mr --count 50 [--start-id 6000] [--seed 42] [--merge] [--dry-run]
python scripts/generate_worksheets.py --type practice --level D3 --language en --count 5
python scripts/generate_worksheets.py --type omr --start-id 990001 --count 100 --merge   # 2 pages each; reuses insert_omr_worksheet logic
python scripts/render_worksheet_pdf.py --worksheet-id 4920 [--out x.pdf]                 # reprint from DB
python scripts/render_worksheet_pdf.py --id 4920 --export-json                         # also write the old JSON shape
```

- `--merge` also writes one combined, print-ready PDF for the batch (like the existing
  `files/pdf/4920to5819.pdf`).
- `--dry-run` renders everything and then rolls back the DB.

## Steps (one small commit each)

| # | Step | Done when |
|---|---|---|
| 9.1 | Move the row-tag encode/decode into `shared/row_tags.py`; vision-service imports it. No behavior change. | Round-trip test over ids 0…35⁵−1 (sampled) and pages 1–2 passes; vision accuracy suite still 10/10 |
| 9.2 | Port the question generators, distractors, `choose_distractors` and the Marathi conversion **unchanged** (only switching to an injected rng) | **Parity test**: with the same seed, the old code and the new code produce identical question JSON for every skill, level A–G and practice level, in en and mr |
| 9.3 | `composition.py`: level distributions plus practice themes; skill difficulty read from `app/data/skills.json` (the file `seed_skills.py` seeds; catalog *order* drives the random draws, and DB row order isn't guaranteed). The service refuses skills missing from the DB | Every level and theme sums to 20; every skill code exists in `skills` |
| 9.4 | `render.py` plus copied assets, `weasyprint` in requirements, and NotoSansDevanagari bundled with an `@font-face` | **Visual parity**: the same id and JSON rendered by the old and new code, rasterized, differ by less than a small pixel threshold (en, mr, regular, OMR p1 and p2) |
| 9.5 | `service.py` plus the CLIs above (insert → render → commit; `--dry-run`, `--merge`, explicit-id collision check) | DB tests: rows, options and `is_correct` match the generated JSON; a render failure leaves no rows; an id collision is refused with nothing written |
| 9.6 | **Scan round-trip golden test**: generate → rasterize the PDF (`pypdfium2`) → vision-service pipeline | Decoded `worksheet_id`, `page_no`, template and question count match; no bubbles marked; covers regular en/mr and OMR pages 1–2 |
| 9.7 | Docker: add the WeasyPrint system libs (`libpango-1.0-0`, `libpangoft2-1.0-0`) to the api-service image; generate inside the container | Marathi PDF from the container is visually identical to the local one (no tofu) |
| 9.8 | Docs: README, ADMIN_HOWTO, PROGRESS; mark the old scripts as superseded | — |

Steps 9.1–9.3 are pure code with no new dependencies, so they can land first. The parity tests in
9.2 and 9.4 import the old modules from the parent repo, the same way the Phase 6 accuracy
bootstrap did. They must never touch the old `.env`'s `DATABASE_URL`, which points at production.
The old generator code is DB-free, so this should be easy to guarantee; assert it in the test setup.

## Port faithfully, flag separately

These look questionable but change the content of what's printed, so the port must not
"fix" them silently. Each needs a decision from the user:

- `create_worksheet_level_distribution` uses `remaining - (len(skill_codes) - i - 2)` (off by one
  compared with the sibling functions' `- 1`), so some skills at a difficulty level can get 0 questions.
- `question_to_marathi` replaces `num1`, then `num2`, using substring `replace`. A question like
  `"3 + 13"` would come out with mixed digits. A scan of 114,000 existing Marathi questions in
  `files/json/` found **no** ASCII digits, so this hasn't happened in practice, but 9.2 adds a test
  asserting it.
- The `mr` templates contain commented-out `{{student_name}}`/`{{worksheet_date}}` placeholders that
  nothing fills. They're harmless; drop them or keep them.

## Open decisions (for the user)

1. **Dashboard "Generate" page**: in scope for this phase, or CLI only for now? The plan assumes CLI
   only, with a dashboard page as a follow-up. That would need auth first, since the admin API is unauthenticated.
2. **Merged print PDF**: should `--merge` be the default?
3. **Ids**: always DB-assigned unless `--start-id` is given? The plan says yes.
4. **JSON file export**: keep it (`export_worksheet_json.py`), or drop it now that the DB is the source of truth?
5. **Retiring the old scripts**: delete them from the repo after 9.8, or leave them frozen?

## Out of scope

- New templates or layouts (template extensibility remains a separate future task).
- Changing question content or difficulty curves.

## Outcome (2026-10-08)

Decisions taken with the plan's defaults, since the open questions weren't answered: CLI only (no
dashboard page); `--merge` is opt-in; ids are DB-assigned unless `--start-id` is given; JSON export
is a `--export-json` flag on `render_worksheet_pdf.py`, not a separate script; the old scripts were
left unmodified, then removed with the rest of the pre-v2 code when v2 moved to the repo root
(still in git history before that commit).

| Step | Result |
|---|---|
| 9.1 | `shared/row_tags.py`; `vision-service/app/vision/tags.py` re-exports it. Round-trip test over 2,007 ids × 3 page variants; equal to the old encoder/decoder on 60k cases (extracted from the old source with `ast`). Vision accuracy suite still 10/10. |
| 9.2–9.3 | `app/generation/{questions,distractors,localization,composition}.py`. **Exact parity with the completely unmodified old generator**: 450/450 worksheets (every homework level and practice level, en and mr, 6–15 seeds each) are identical, and 0/450 match when the seeds are shifted, so the test is sensitive. Independent property test: in 1,350 sheets (27,000 questions), the answer recomputed from the question text always equals the option `correct_option` points to; options are unique; Marathi has no ASCII digits. |
| 9.4 | `app/generation/render.py`, assets in `api-service/assets/worksheets/` (870 KB, down from 7 MB, since only the 4 corner tags are copied). **Pixel-identical** (0.00000% of pixels differ at 100 dpi) to the old renderer for homework en/mr, practice mr, and OMR pages 1–2, and to three sheets that were actually printed (`files/pdf/4920`, `5500`, `5819`, rendered from their `files/json`). |
| 9.5 | `app/generation/service.py`, `scripts/generate_worksheets.py`, `scripts/render_worksheet_pdf.py`. 9 DB tests: seeds/answer keys stored, OMR pages, DB-assigned id is the printed one, taken ids refused, render failure and write failure both leave no rows and no files, dry run, reprint. CLI smoke-tested by hand, then cleaned up. |
| 9.6 | `vision-service/tests/test_generated_sheets.py` scans 5 generated sheets (150 dpi; including the largest encodable id, 52,521,874) with the real pipeline: id, page, template and every question decode, with no false marks. `api-service/tests/test_generation_scan_fixtures.py` keeps those PNGs identical to current generator output (`UPDATE_SCAN_FIXTURES=1` regenerates them). Ad hoc, not a test: a filled bubble on a generated homework sheet (Q3 C) and on OMR page 2 (Q50 B) was read correctly. |
| 9.7 | `requirements.txt`: `weasyprint==68.1`, `pypdfium2`. Dockerfile: Pango, `fonts-dejavu-core`, bundled Noto Sans Devanagari. **Not built or run.** |
| 9.8 | README, ADMIN_HOWTO, PROGRESS updated. |

Deviations from a byte-for-byte port, none of which change any output (the parity tests pass against the unmodified old code):
- Randomness is an injected `random.Random`, not the global module.
- The distractor functions no longer shuffle a shared mutable default `offsets` list. In practice this never changed an output.
- Marathi conversion translates every digit in the string instead of doing two substring replaces. That avoids the latent `"3 + 13"` → `"३ + 1३"` bug, which no current generator can trigger.
- The templates name their fonts explicitly. Previously `body` had no `font-family` at all, the `@font-face` pointed at a file that didn't exist, and the faces came from fontconfig defaults and fallback. The output is still pixel-identical.

Notes:
- The OMR header is hardcoded as "Navodaya Practice Exam OMR Sheet", as before.
- `template_mr.html` was never selected by the old code: Marathi sheets print the English instructions. The file wasn't copied.
- Rolled-back batches leave gaps in the worksheet id sequence, because Postgres sequences aren't transactional.

Still to do:
1. Build the api-service image and generate a Marathi PDF inside it. Then rasterize it and compare against a local render: the parity helpers in `tests/test_generation_render.py` work on any two PDFs.
2. The two "port faithfully, flag separately" content questions above still need a decision.
