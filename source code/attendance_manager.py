import csv
import os
from datetime import datetime

CSV_FILE = "Attendance.csv"


def initialize_csv():
    """Step A: create Attendance.csv with headers if it doesn't exist yet.
    Call this once, at the start of main.py, before the webcam loop starts."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])
        print(f"Created new {CSV_FILE}")


def is_already_marked_today(name, today_date):
    """Step B: check every existing row to see if this person already
    has an entry for today's date. Prevents duplicate attendance marks
    if the same face stays in frame for multiple loop iterations."""
    if not os.path.exists(CSV_FILE):
        return False

    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == today_date:
                return True
    return False


def mark_attendance(name):
    """Step C: the main function main.py will call for every recognized face.
    Returns True if a new row was written, False if the person was already
    marked today (so main.py can optionally show a different message)."""
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    if is_already_marked_today(name, today_date):
        return False  # already marked today, do nothing

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today_date, current_time])

    print(f"Attendance marked: {name} at {current_time}")
    return True


# Quick standalone test — run this file directly (python attendance_manager.py)
# to confirm the CSV logic works before wiring it into main.py.
if __name__ == "__main__":
    initialize_csv()
    print("Testing mark_attendance('TestUser') twice in a row...")
    first_call = mark_attendance("TestUser")
    second_call = mark_attendance("TestUser")
    print(f"First call wrote a new row: {first_call}")
    print(f"Second call wrote a new row (should be False): {second_call}")
    print(f"Check {CSV_FILE} — 'TestUser' should appear only ONCE.")