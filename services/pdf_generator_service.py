from weasyprint import HTML, CSS
import json
from config import SETTINGS
from services.image_service import encode_worksheet_id

ORIENTATION_ID = SETTINGS.ORIENTATION_ID
TAGS_PATH = SETTINGS.TAGS_PATH

# opens a json worksheet file
def open_worksheet(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    return questions

# from a worksheet ID (check database), get the tags to be inserted
def getTagNumbers(id):
    idTags = encode_worksheet_id(id)
    idTags.insert(0, ORIENTATION_ID)

    return idTags

# only for corner tags (36h11)
def generate_tags_html(tag_ids, tags_folder_path=TAGS_PATH):
    tag_urls = [f"{tags_folder_path}/36h11/tag36_11_{str(id).zfill(5)}.svg" for id in tag_ids]
    print(f"tag urls: {tag_urls}")
    tags_html = ""

    print(f"Adding tags {tag_urls}")
    tags_html += f'<div class="marker top-left" style="background-image: url({tag_urls[0]})"></div>\n'
    tags_html += f'<div class="marker top-right" style="background-image: url({tag_urls[1]})""></div>\n'
    tags_html += f'<div class="marker bottom-left" style="background-image: url({tag_urls[2]})"></div>\n'
    tags_html += f'<div class="marker bottom-right" style="background-image: url({tag_urls[3]})"></div>\n'

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

def generate_questions_html(questions, tags_folder_path=TAGS_PATH):
    rows_html = ""
    for i in range(0, len(questions), 2):
        q1 = questions[i]
        q2 = questions[i+1] if i+1 < len(questions) else None
        rows_html += "<tr>\n"
        # add one marker per row
        rows_html += "<td class='row-marker'>\n"

        # tag url for 25h9
        tag_url = f"{tags_folder_path}/25h9/tag25_09_{str((i // 2) + 1).zfill(5)}.svg"

        # tag url for 36h11 tags number 10-20
        # tag_url = f"tags/36h11/tag36_11_{str((i // 2) + 10).zfill(5)}.svg"

        print(f"Adding {tag_url}")
        rows_html += f"<div class='marker' style='background-image: url({tag_url})'></div>\n"
        rows_html += "</td>\n"
        rows_html += f"{generate_question_box(q1, i+1)}\n"
        rows_html += f"{generate_question_box(q2, i+2) if q2 else ''}\n"
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