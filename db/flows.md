## Flows
add a new worksheet to the database
- generate worksheet json using generator
- assign the worksheet a worksheet_id
- add field worksheet_id to the json
- create new entry in worksheets - requires worksheet_id, worksheet_level, is_test, max_score, lang, worksheet_json
- read every question from the worksheet json and assign it a question_id
- add every question to the questions database

generate pdf
- retrieve json from worksheets - requires worksheet_id
- call pdf_generator module
- receive a submission from a student
- check whether from_number is in users or not.

process the worksheet
- if worksheet is not detected, abandon processing and save as failed attempt.
- retrieve the answer key from the database (worksheet's json - "ans_key")
- extract worksheet_id from worksheet
- check worksheet and get results_json, score, from_number
- insert record into submissions - requires student_id, worksheet_id, score, from_number, answers_json
- process answers_json into attempt records, and insert all - requires student_id, submission_id, question_id, worksheet_id, is_correct, skill_code

manually submit a worksheet which isn't detecting or mistake has been made
- send a picture from teacher role (from_number)
- system sends back an empty template with worksheet id, question-wise option selected to be filled
(later) system tries to detect what it can and fills the template with that before sending it
- user sends the template back
- query submissions with the worksheet_id mentioned in the template
- if already exists, overwrite it (overwrite all the attempt entries as well)
- otherwise, follow same remaining flow as "receive a submission from a student"
- generate a checked image with the text score on the top-right (regardless of worksheet detection)