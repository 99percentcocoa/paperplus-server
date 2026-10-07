"""Template layout registry — ported as-is from the old repo's template_layouts.py.
Kept code-based/low-priority-extensibility per the redevelopment plan; not DB-driven.
"""

from dataclasses import dataclass, field
from typing import Optional

from app.vision.geometry import ROI as ROIBox

ROITuple = tuple[int, int, int, int]


@dataclass(frozen=True)
class TemplateLayout:
    name: str
    num_questions: int = 20
    num_row_tags: int = 10
    left_question_roi: ROITuple = (85, -40, 485, 90)
    right_question_roi: ROITuple = (620, -40, 485, 90)
    roll_number_roi: Optional[ROITuple] = (420, 1660, 850, 1754)
    question_roi_columns: tuple[ROITuple, ...] = ()
    handwritten_fields: dict[str, Optional[ROITuple]] = field(default_factory=dict)

    def __post_init__(self):
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

TEMPLATE_LAYOUTS: dict[str, TemplateLayout] = {
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
    if template_name is None:
        return LEGACY_LAYOUT
    normalized = template_name.strip().lower()
    return TEMPLATE_LAYOUTS.get(normalized, LEGACY_LAYOUT)


def get_question_rois_for_template(template_name: str | None = None) -> list[ROITuple]:
    return list(get_template_layout(template_name).question_roi_columns)


def get_template_num_questions(template_name: str | None = None) -> int:
    return get_template_layout(template_name).num_questions


def get_template_row_tag_count(template_name: str | None = None) -> int:
    return get_template_layout(template_name).num_row_tags


def get_handwritten_field_roi(template_name: str | None, field_name: str) -> Optional[ROITuple]:
    return get_template_layout(template_name).handwritten_fields.get(field_name)


def infer_template_name_from_row_metadata(row_metadata: dict | None, template_hint: str | None = None) -> str:
    """Resolve the template purely from data available to a stateless vision-service.

    Unlike the old system, this never consults the DB for a worksheet's stored template
    name (vision-service has no DB access by design). If the caller (api-service) already
    knows the expected template, pass it as `template_hint` and it wins; otherwise the row-tag
    format signal (omr_v2 -> basic_omr) is used, matching the old system's primary/most common
    inference path.
    """
    if template_hint:
        return template_hint.strip().lower()
    if row_metadata and row_metadata.get("format") == "omr_v2":
        return "basic_omr"
    return "regular"


def get_roi_coordinates(row_detection_centers: list[tuple[float, float]], template_name: str | None = None) -> list[ROIBox]:
    """Map row-tag anchor centers to absolute question ROIs, template-aware."""
    roi_coordinates: list[ROIBox] = []
    question_rois = get_question_rois_for_template(template_name)

    for anchor_x, anchor_y in row_detection_centers:
        for (rx, ry, rw, rh) in question_rois:
            x1 = int(anchor_x + rx)
            y1 = int(anchor_y + ry)
            x2 = int(x1 + rw)
            y2 = int(y1 + rh)
            roi_coordinates.append(ROIBox(x1, y1, x2, y2))

    return roi_coordinates
