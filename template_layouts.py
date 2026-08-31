from dataclasses import dataclass, field
from typing import Optional, Tuple

ROI = Tuple[int, int, int, int]


@dataclass(frozen=True)
class TemplateLayout:
    name: str
    num_questions: int = 20
    num_row_tags: int = 10
    default_num_questions: int = 20
    left_question_roi: ROI = (85, -40, 485, 90)
    right_question_roi: ROI = (620, -40, 485, 90)
    roll_number_roi: Optional[ROI] = (420, 1660, 850, 1754)
    question_roi_columns: tuple[ROI, ...] = ()
    handwritten_fields: dict[str, Optional[ROI]] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "default_num_questions", self.num_questions)
        object.__setattr__(
            self,
            "question_roi_columns",
            tuple(self.question_roi_columns or (self.left_question_roi, self.right_question_roi)),
        )


LEGACY_LAYOUT = TemplateLayout(
    name="regular",
    num_questions=20,
    num_row_tags=10,
    left_question_roi=(85, -40, 485, 90),
    right_question_roi=(620, -40, 485, 90),
    roll_number_roi=(420, 1660, 850, 1754),
    question_roi_columns=((85, -40, 485, 90), (620, -40, 485, 90)),
    handwritten_fields={
        "roll_number": (420, 1660, 850, 1754),
        "question_paper_code": None,
    },
)

TEMPLATE_LAYOUTS = {
    "regular": LEGACY_LAYOUT,
    "basic_omr": TemplateLayout(
        name="basic_omr",
        num_questions=20,
        num_row_tags=10,
        left_question_roi=(85, -40, 485, 90),
        right_question_roi=(620, -40, 485, 90),
        roll_number_roi=(420, 1660, 850, 1754),
        question_roi_columns=((50, -40, 365, 90), (430, -40, 365, 90), (810, -40, 365, 90)),
        handwritten_fields={
            "roll_number": (820, 1650, 1150, 1754),
            "question_paper_code": (100, 1650, 320, 1754),
        },
    ),
}


def get_template_layout(template_name: str | None = None) -> TemplateLayout:
    """Return the layout config for the selected template name."""
    if template_name is None:
        return LEGACY_LAYOUT

    normalized = template_name.strip().lower()
    return TEMPLATE_LAYOUTS.get(normalized, LEGACY_LAYOUT)


def get_question_rois_for_template(template_name: str | None = None) -> list[ROI]:
    return list(get_template_layout(template_name).question_roi_columns)


def get_template_num_questions(template_name: str | None = None) -> int:
    return get_template_layout(template_name).num_questions


def get_template_row_tag_count(template_name: str | None = None) -> int:
    return get_template_layout(template_name).num_row_tags


def get_handwritten_field_roi(template_name: str | None, field_name: str) -> Optional[ROI]:
    return get_template_layout(template_name).handwritten_fields.get(field_name)
