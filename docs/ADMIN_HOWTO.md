# Admin how-to: students, OMR worksheets, answer keys, uploading scans

Scripts run from `api-service/` with its venv active (`source .venv/bin/activate`),
or inside the container: `docker compose exec api-service python scripts/<script>.py ...`.
`<host>` below is wherever api-service is reachable, e.g. `http://localhost:8000` locally or the
server's public URL in production.

Projects: `paperplus` and `navodaya`. Each has its own dashboard at `<host>/admin/<project>/`.
Share that link with the people who should see that project.

## Add schools and students

Put the students in a CSV with a name column (header containing "name") and, optionally, an ID
column (header containing "id" or "roll"). One CSV = one school.

```bash
# PaperPlus (the default project)
python scripts/import_students_from_csv.py students.csv --school-code KGP

# Navodaya
python scripts/import_students_from_csv.py navodaya_batch1.csv --school-code NAV1 --project navodaya
```

- The school is created if it doesn't exist (its name = the code). A school belongs to one
  project. Reusing a PaperPlus school code with `--project navodaya` is refused.
- **Student IDs are unique across both projects.** This ID is what's written on the sheet and
  read by OCR. If the CSV has no ID column, IDs are assigned from the next free 4-digit number,
  or from `--start-id 5000`. If an ID already belongs to a student in the other project, the
  whole import is refused and nothing is written.
- Re-running the same CSV is safe: existing students in the same project are left as they are.

## Generate homework and practice worksheets

One command writes the questions to the database *and* the printable PDFs, so every printed
sheet's ID is one the database knows:

```bash
python scripts/generate_worksheets.py --type homework --level A --language mr --count 50
python scripts/generate_worksheets.py --type practice --level D3 --language en --count 5
python scripts/generate_worksheets.py --type homework --level C --count 50 --merge      # + one print file
python scripts/generate_worksheets.py --type homework --level C --count 50 --dry-run    # check first
```

- Levels: homework `A`–`G`; practice is a theme plus a level, `A1`–`A5`, `S1`–`S5`, `M1`–`M5`,
  `D1`–`D5` (Addition, Subtraction, Multiplication, Division).
- IDs: the database picks the next free IDs. To choose them, add `--start-id 6000`; if any ID
  in the range is taken, nothing is written.
- PDFs go to `files/worksheets/` (or `--out-dir`), one per sheet, named
  `<id>_<lang>_<type>_<LEVEL>.pdf`. `--merge` also writes one combined `..._print.pdf` for the
  print shop.
- Run `python scripts/seed_skills.py` once before the first batch on a new database.
- Every sheet's random seed is saved with it, so a sheet can be regenerated identically.
  `--seed N` fixes the seeds (sheet *i* gets seed N + *i*).

**In docker**, write the PDFs to a folder on the host:

```bash
docker compose run --rm -v "$PWD/print:/print" api-service \
  python scripts/generate_worksheets.py --type homework --level A --language mr --count 50 --merge --out-dir /print
```

## Add a new OMR worksheet

```bash
python scripts/generate_worksheets.py --type omr --start-id 5001 --count 100 --merge
```

Each sheet gets 78 questions on 2 pages (Q1–39 and Q40–78) and two PDFs (`..._page1.pdf`,
`..._page2.pdf`). Use `--questions 39` for a 1-page sheet. Then add the answer key for each
question-paper code (next section).

**Sheets already printed by the pre-v2 system** only need
registering, using the IDs printed on them:

```bash
python scripts/insert_omr_worksheet.py --start-id 5001 --count 100 --dry-run   # check first
python scripts/insert_omr_worksheet.py --start-id 5001 --count 100
```

IDs that are already OMR worksheets are skipped, so re-running a range is safe. If any ID in the
range is already a homework/practice worksheet, nothing is inserted and the clashing IDs are
listed. Don't use `insert_single_worksheet.py --json-file blank_omr.json` for OMR sheets. That
file only has 39 questions, so page 2 of the printed sheet couldn't be graded.

## Reprint worksheets

Any worksheet in the database can be printed again from its stored questions. This includes
sheets made before v2, as long as they were inserted or migrated:

```bash
python scripts/render_worksheet_pdf.py --id 4920
python scripts/render_worksheet_pdf.py --start-id 4920 --count 900 --merge
python scripts/render_worksheet_pdf.py --id 4920 --export-json    # also write the JSON
```

Worksheets aren't tied to a project. The same OMR worksheet ID can be used by PaperPlus and
Navodaya students.

## Add an answer key for a question-paper set (code A–F)

The letter written in the sheet's question-paper-code box selects the answer key. One key per
code works for every OMR worksheet:

```bash
python scripts/insert_code_answer_key.py --code D \
  --answer-key A,C,B,D,A,B,C,D,...        # one letter per question, Q1 first (78 for a 2-page sheet)
```

Running it again for the same code replaces that key. If one particular worksheet needs a
different key for a code, `insert_question_paper_variant.py --worksheet-id 5001 --code D
--answer-key ...` overrides it for that worksheet only. A sheet scanned with a code that has no
key is not graded. It shows up under "Failed scans" as missing an answer key.

## Send a scan without WhatsApp, optionally giving the handwritten fields

**From the dashboard:** open the project's dashboard, then **Upload** in the top bar
(`<host>/admin/<project>/#/upload`). Choose one or more photos, optionally type the roll number
and/or question-paper code, and press *Upload and grade*. Each photo gets a result row with the
score (linking to the submission) or the reason it failed (linking to the failed scan). A typed
roll number/code applies to **every** photo in that upload: use it for the pages of one
student's sheet, or leave it blank to read it from each photo.

**From a terminal (curl):** same endpoint the Upload page uses.

```bash
# Everything read from the photo, like WhatsApp:
curl --data-binary @scan.jpg "<host>/api/admin/projects/navodaya/scans"

# Give the student ID and/or the question-paper code (skips reading them from the photo):
curl --data-binary @scan.jpg "<host>/api/admin/projects/navodaya/scans?roll_number=5012&question_paper_code=D"
```

- Use `--data-binary @file` (not `-d` or `-F`). JPEG, PNG or WebP, up to 20 MB.
- Fields you leave out are read from the photo as usual. A field you give replaces OCR for that
  field, which also rescues sheets whose handwriting can't be read.
- The student must belong to the project in the URL. Otherwise you get a `400` and nothing is
  graded.
- It's graded exactly like a WhatsApp scan: submission, score, mastery, checked image. Nothing is
  sent over WhatsApp. The reply that would have been sent comes back in the JSON response
  (`replies`), with `score`, `submission_id`, `checked_image_url` (relative to `<host>`), or, on
  failure, `error_reason` and a `review_id` that appears on that project's Failed scans page.
- Add `&from_number=+91...` to record a phone number on the scan (still nothing is sent).
- Grading takes a few seconds per photo. To upload a folder:
  ```bash
  for f in scans/*.jpg; do curl -s --data-binary @"$f" "<host>/api/admin/projects/navodaya/scans"; echo; done
  ```
- Through a reverse proxy, the proxy must allow bodies of several MB
  (`client_max_body_size 20m;` in nginx; the default of 1 MB rejects phone photos with 413).
