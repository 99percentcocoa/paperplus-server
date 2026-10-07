"""Draws the checked-image (correct/incorrect annotation) sent to students after grading.

Ported from the old system's services/grading_service.py:check_worksheet (per-question ✔/✘/?
overlay) and services/image_service.py:save_checked/make_circle_mark (score badge). Uses PIL,
not cv2, because ✔/✘ are Unicode glyphs cv2.putText cannot render.

This only runs after grading, so it needs both vision-service's pixel ROI boxes (carried on each
QuestionMark, see shared/contracts.py) and api-service's correctness (from grade_marks) -- neither
service alone has both.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
SYMBOL_FONT_PATH = FONTS_DIR / "NotoSansSymbols2-Regular.ttf"
BOLD_FONT_PATH = FONTS_DIR / "NotoSans-Bold.ttf"

GREEN = (0, 127, 0)
RED = (255, 86, 86)
DARK_BLUE = (10, 20, 120, 255)


def draw_checked_image(
    dewarped_image_path: str,
    question_marks: list,
    answers_payload: list[dict],
    score: int,
    total: int,
    output_path: str,
) -> str:
    """question_marks: shared.contracts.QuestionMark instances (need roi_x1/y1/x2/y2 +
    question_index). answers_payload: grade_marks() output ({question_index, selected_option,
    is_correct} dicts). Returns output_path for convenience (also written to disk).
    """
    image = Image.open(dewarped_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    symbol_font = ImageFont.truetype(str(SYMBOL_FONT_PATH), 60)

    answers_by_index = {a["question_index"]: a for a in answers_payload}

    for mark in question_marks:
        answer = answers_by_index.get(mark.question_index)
        if answer is None:
            continue

        x1, y1, x2, y2 = mark.roi_x1, mark.roi_y1, mark.roi_x2, mark.roi_y2
        width = x2 - x1

        if answer["is_correct"]:
            color, glyph = GREEN, "✔"  # ✔
        elif not answer["selected_option"]:
            color, glyph = RED, "?"
        else:
            color, glyph = RED, "✘"  # ✘

        draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=color)
        draw.text((x1 + width - 5, y1 - 5), glyph, fill=color, font=symbol_font)

    circle = _make_circle_mark(score, total)
    image.paste(circle, (100, 50), circle)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _make_circle_mark(obtained: int, total: int, diameter: int = 150) -> Image.Image:
    img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.ellipse([(5, 5), (diameter - 5, diameter - 5)], outline=DARK_BLUE, width=7)

    center_y = diameter // 2
    draw.line([(20, center_y), (diameter - 20, center_y)], fill=DARK_BLUE, width=7)

    font = ImageFont.truetype(str(BOLD_FONT_PATH), 50)

    top_text = str(obtained)
    bbox = draw.textbbox((0, 0), top_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((diameter - tw) // 2, center_y - th - 30), top_text, fill=DARK_BLUE, font=font)

    bottom_text = str(total)
    bbox2 = draw.textbbox((0, 0), bottom_text, font=font)
    tw2 = bbox2[2] - bbox2[0]
    draw.text(((diameter - tw2) // 2, center_y), bottom_text, fill=DARK_BLUE, font=font)

    return img
