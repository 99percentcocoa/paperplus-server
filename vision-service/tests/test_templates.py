from app.vision.templates import (
    get_handwritten_field_roi,
    get_question_rois_for_template,
    get_roi_coordinates,
    get_template_layout,
    get_template_num_questions,
    get_template_row_tag_count,
    infer_template_name_from_row_metadata,
)


def test_regular_template_defaults():
    layout = get_template_layout(None)
    assert layout.name == "regular"
    assert layout.question_roi_columns == ((85, -40, 485, 90), (620, -40, 485, 90))


def test_unknown_template_falls_back_to_regular():
    layout = get_template_layout("does-not-exist")
    assert layout.name == "regular"


def test_basic_omr_has_three_columns_and_paper_code_field():
    rois = get_question_rois_for_template("basic_omr")
    assert len(rois) == 3
    assert get_handwritten_field_roi("basic_omr", "question_paper_code") == (100, 1650, 320, 1754)


def test_regular_has_no_paper_code_field():
    assert get_handwritten_field_roi("regular", "question_paper_code") is None


def test_template_num_questions_and_row_tag_count():
    assert get_template_num_questions("basic_omr") == 20
    assert get_template_row_tag_count("regular") == 10


def test_infer_template_hint_wins_over_row_metadata():
    assert infer_template_name_from_row_metadata({"format": "omr_v2"}, template_hint="regular") == "regular"


def test_infer_template_from_omr_v2_format():
    assert infer_template_name_from_row_metadata({"format": "omr_v2"}) == "basic_omr"


def test_infer_template_defaults_to_regular():
    assert infer_template_name_from_row_metadata({"format": "legacy"}) == "regular"
    assert infer_template_name_from_row_metadata(None) == "regular"


def test_get_roi_coordinates_offsets_from_anchor():
    rois = get_roi_coordinates([(100.0, 200.0)], "regular")
    assert len(rois) == 2
    first, second = rois
    assert (first.x1, first.y1, first.x2, first.y2) == (185, 160, 670, 250)
    assert (second.x1, second.y1, second.x2, second.y2) == (720, 160, 1205, 250)
