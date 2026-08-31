import logging
import json
from config import SETTINGS

ORIENTATION_ID = SETTINGS.ORIENTATION_ID
TAGS_PATH = SETTINGS.TAGS_PATH

logger = logging.getLogger(__name__)

# opens a json worksheet file
def open_worksheet(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    return questions

# from a worksheet ID (check database), get the tags to be inserted
# def getTagNumbers(id):
#     idTags = encode_worksheet_id(id)
#     idTags.insert(0, ORIENTATION_ID)

#     return idTags

# only for corner tags (36h11)
def generate_tags_html(tag_ids, tags_folder_path=TAGS_PATH):
    tag_urls = [f"{tags_folder_path}/36h11/tag36_11_{str(id).zfill(5)}.svg" for id in tag_ids]
    logger.info("Corner tag URLs: %s", tag_urls)
    tags_html = ""

    tags_html += f'<img class="marker top-left" src="{tag_urls[0]}" alt="tag 0" />\n'
    tags_html += f'<img class="marker top-right" src="{tag_urls[1]}" alt="tag 1" />\n'
    tags_html += f'<img class="marker bottom-left" src="{tag_urls[2]}" alt="tag 2" />\n'
    tags_html += f'<img class="marker bottom-right" src="{tag_urls[3]}" alt="tag 3" />\n'

    logger.info("Added corner tags: %s", tag_urls)
    return tags_html

# corner tags using cctag
def generate_cctag_html(tag_ids=[0,1,2,3],tags_folder_path=TAGS_PATH):
    tag_urls = [f"{tags_folder_path}/cctags/{str(id).zfill(4)}.svg" for id in tag_ids]
    logger.info("CC tag URLs: %s", tag_urls)
    tags_html = ""

    logger.info("Adding CC tags: %s", tag_urls)
    tags_html += f'<div class="marker top-left"><img src="{tag_urls[0]}" style="width:100%;height:100%;" /></div>\n'
    tags_html += f'<div class="marker top-right"><img src="{tag_urls[1]}" style="width:100%;height:100%;" /></div>\n'
    tags_html += f'<div class="marker bottom-left"><img src="{tag_urls[2]}" style="width:100%;height:100%;" /></div>\n'
    tags_html += f'<div class="marker bottom-right"><img src="{tag_urls[3]}" style="width:100%;height:100%;" /></div>\n'

    return tags_html

def generate_question_box(question, q_no):
    option_html = ""

    option_html += "<td class='question_td'>\n <div class='question'>\n"
    option_html += f"<p>{q_no}. {question['question_text']}</p>"
    option_html += "<table class='options-table'>\n <tr>"
    option_html += f"<td><div class='circle'></div>A. {question['options'][0]}</td>"
    option_html += f"<td><div class='circle'></div>B. {question['options'][1]}</td>"
    option_html += f"<td><div class='circle'></div>C. {question['options'][2]}</td>"
    option_html += f"<td><div class='circle'></div>D. {question['options'][3]}</td>"
    option_html += "</tr>\n </table>"
    option_html += "</div>\n </td>"

    return option_html


def generate_basic_omr_questions_html(
    worksheet_id: int,
    question_count: int = 20,
    tags_folder_path: str = TAGS_PATH,
    page_no: int = 1,
    first_question_index: int = 1,
):
    """Build a blank answer-grid for the OMR template without question text or answer keys.

    When `page_no` is greater than 1 or `first_question_index` is greater than 1,
    the sheet continues numbering from that starting question index. This allows
    multi-page OMR sheets such as page 2 starting at question 40.
    """
    from services.image_service import worksheet_id_to_rows

    question_count = max(0, int(question_count or 0))
    page_no = max(1, int(page_no or 1))
    first_question_index = max(1, int(first_question_index or 1))
    if question_count == 0:
        return ""

    row_tags = worksheet_id_to_rows(
        worksheet_id,
        page_no=page_no if page_no is not None else 1,
        first_question_index=first_question_index if first_question_index is not None else 1,
    )
    logger.info("Generated row tags for worksheet %s: %s", worksheet_id, row_tags)
    rows_html = ""
    for row_index in range(0, question_count, 3):
        row_num = row_index // 3
        row_tag_id = row_tags[row_num % len(row_tags)] if row_tags else 0
        rows_html += "<tr>\n"

        rows_html += "<td class='row-marker'>\n"
        tag_url = f"{tags_folder_path}/25h9/tag25_09_{str(row_tag_id).zfill(5)}.svg"
        logger.info("Adding row tag URL: %s", tag_url)
        rows_html += f"<img class='marker' src='{tag_url}' alt='row tag' />\n"
        rows_html += "</td>\n"

        for offset in range(3):
            q_no = first_question_index + row_index + offset
            if q_no - first_question_index >= question_count:
                break

            rows_html += "<td class='question_td'>\n <div class='question'>\n"
            rows_html += f"<p>{q_no}.</p>"
            rows_html += "<table class='options-table'>\n <tr>"
            rows_html += "<td><div class='circle'></div>A</td>"
            rows_html += "<td><div class='circle'></div>B</td>"
            rows_html += "<td><div class='circle'></div>C</td>"
            rows_html += "<td><div class='circle'></div>D</td>"
            rows_html += "</tr>\n </table>"
            rows_html += "</div>\n </td>\n"

        rows_html += "</tr>\n"

    return rows_html

# now embed the ID in the question tags
def generate_questions_html(worksheet_id, questions, tags_folder_path=TAGS_PATH):
    from services.image_service import worksheet_id_to_rows

    row_tags = worksheet_id_to_rows(worksheet_id)
    logger.info("Row tags: %s", row_tags)
    rows_html = ""

    if len(questions) != 20:
        raise ValueError("Expected exactly 20 questions, got {}".format(len(questions)))
    
    for i in range(0, len(questions), 2):
        q1 = questions[i]
        q2 = questions[i+1] if i+1 < len(questions) else None
        rows_html += "<tr>\n"
        row_num = i // 2
        row_tag_id = row_tags[row_num % len(row_tags)] if row_tags else 0
        # add two markers per row

        rows_html += "<td class='row-marker'>\n"
        tag_url = f"{tags_folder_path}/25h9/tag25_09_{str(row_tag_id).zfill(5)}.svg"
        logger.info("Adding %s", tag_url)
        rows_html += f"<div class='marker' style='background-image: url({tag_url})'></div>\n"
        rows_html += "</td>\n"

        rows_html += f"{generate_question_box(q1, i+1)}\n"
        rows_html += f"{generate_question_box(q2, i+2) if q2 else ''}\n"

        #second marker
        # rows_html += "<td class='row-marker'>\n"
        # tag_url = f"{tags_folder_path}/25h9/tag25_09_{str(row_tags[row_num]).zfill(5)}.svg"
        # print(f"Adding {tag_url}")
        # rows_html += f"<div class='marker' style='background-image: url({tag_url})'></div>\n"
        # rows_html += "</td>\n"

        rows_html += "</tr>\n"
    
    return rows_html

# if __name__ == "__main__":

#     worksheet_id = 2

#     with open("template.html", "r", encoding="utf-8") as f:
#         template_html = f.read()

#     questions = open_worksheet('marathi.json')
#     questions_html = generate_questions_html(questions)
#     tags_html = generate_tags_html(getTagNumbers(worksheet_id))
#     final_html = template_html.replace("{{tags_html}}", tags_html).replace("{{questions}}", questions_html)

#     HTML(string=final_html, base_url=".").write_pdf("output.pdf")