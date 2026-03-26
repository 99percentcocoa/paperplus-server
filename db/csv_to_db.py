import sys
import csv
import psycopg
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SETTINGS

DATABASE_URL = SETTINGS.DATABASE_URL
CSV_PATH = Path(__file__).resolve().parent.parent / "files" / "csv"
SCHOOLS_FILE = CSV_PATH / "paperplus_schools.csv"
STUDENTS_FILE = CSV_PATH / "paperplus_students.csv"

conn = psycopg.connect(DATABASE_URL)
cur = conn.cursor()

def import_schools():

    with open(SCHOOLS_FILE, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            school_name = row['School Name']
            school_code = row['School Code']
            
            cur.execute("""INSERT INTO schools (school_code, school_name) 
                        VALUES (%s, %s)
                        ON CONFLICT (school_code) DO NOTHING""", 
                        (school_code, school_name))

    print("Schools imported.")

def import_students():

    # load school mapping
    cur.execute("SELECT school_code, school_name FROM schools")
    school_mapping = {
        row[1]: row[0] for row in cur.fetchall()
    }

    with open(STUDENTS_FILE, mode='r') as file:
        reader = csv.DictReader(file)

        for row in islice(reader, 1000):
            student_name = row['Student Name']
            student_id = row['Student ID']
            school_name = row['School'].strip()

            school_id = school_mapping.get(school_name)

            if not school_id:
                print(f"Warning: School '{school_name}' not found for student '{student_name}'. Skipping.")
                continue

            cur.execute("""
                INSERT INTO STUDENTS (student_id, student_name, student_school_code, current_level)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (student_id) DO NOTHING
            """, (student_id, student_name, school_id, "A"))
    
    print("Students imported.")

if __name__ == "__main__":
    import_schools()
    import_students()
    conn.commit()
    cur.close()
    conn.close()