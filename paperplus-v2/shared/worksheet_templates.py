"""Per-template question-per-page counts, shared between api-service (worksheet insertion,
computing page_count/worksheet_pages) and, potentially, vision-service in future.

Deliberately does NOT include ROI/geometry -- that stays code-only inside vision-service's own
TEMPLATE_LAYOUTS (see app/vision/templates.py), since vision-service has no DB access by design
and geometry is effectively a compiled constant tied to the printed PDF layout, not runtime data.
This module only carries the one fact api-service needs that vision-service's own TemplateLayout
doesn't reliably expose today (its `num_questions` field is stale/unused for basic_omr).
"""

QUESTIONS_PER_PAGE: dict[str, int] = {
    "regular": 20,
    "basic_omr": 39,
}
