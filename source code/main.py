import cv2
import face_recognition

from face_recognizer import load_known_faces  # Member 1's function, reused as-is
from attendance_manager import initialize_csv, mark_attendance  # your functions

# Same tuning values Member 1 used, kept consistent for identical accuracy/speed.
MATCH_TOLERANCE = 0.6
FRAME_RESIZE_SCALE = 0.25


def main():
    # Step A: load the encodings your teammate generated.
    known_encodings, known_names = load_known_faces()
    if known_encodings is None:
        print("encodings.pickle not found. Ask your teammate to run encode_faces.py first.")
        return

    print(f"Loaded {len(known_encodings)} known face encoding(s).")

    # Step B: make sure Attendance.csv exists before we start.
    initialize_csv()

    # Step C: open the webcam.
    # cv2.CAP_DSHOW forces the DirectShow backend, which is far more
    # reliable than OpenCV's default (MSMF) on many Windows machines.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback: try the default backend in case DSHOW isn't available.
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera connection/permissions.")
        return

    print("Attendance system started. Press ESC to quit.\n")

    # Keep track of names already marked THIS SESSION so we don't call
    # mark_attendance() (and hit the CSV) every single frame for the same
    # person — only once per person is enough, is_already_marked_today()
    # in attendance_manager.py double-checks this too.
    marked_this_session = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Step D: shrink frame for faster processing, convert BGR -> RGB.
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Step E: detect every face + compute its encoding.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Step F: compare each detected face against known encodings.
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=MATCH_TOLERANCE
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            face_names.append(name)

            # Step G: THIS is your core contribution — mark attendance
            # the moment a known face is recognized.
            if name != "Unknown" and name not in marked_this_session:
                was_marked = mark_attendance(name)
                if was_marked:
                    marked_this_session.add(name)

        # Step H: draw boxes + labels back on the full-size frame.
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            scale = int(1 / FRAME_RESIZE_SCALE)
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1,
            )

        cv2.imshow("Facial Attendance System", frame)

        # Step I: ESC key (27) exits.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Step J: cleanup.
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended. Check Attendance.csv for today's records.")


if __name__ == "__main__":
    main()