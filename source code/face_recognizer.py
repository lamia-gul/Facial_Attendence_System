import pickle
import cv2
import face_recognition

ENCODINGS_FILE = "encodings.pickle"

# How strict the match needs to be. Lower = stricter. 0.6 is the
# face_recognition library's recommended default.
MATCH_TOLERANCE = 0.6

# Resize frames before processing to make recognition faster.
# 0.25 = process at 1/4 resolution, then scale coordinates back up.
FRAME_RESIZE_SCALE = 0.25


def load_known_faces():
    """Step A: load the pickle file created by encode_faces.py."""
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    except FileNotFoundError:
        return None, None


def main():
    known_encodings, known_names = load_known_faces()
    if known_encodings is None:
        print(f"'{ENCODINGS_FILE}' not found. Run encode_faces.py first.")
        return

    print(f"Loaded {len(known_encodings)} known face encoding(s).")

    # Step B: open the webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera connection/permissions.")
        return

    print("Recognition started. Press ESC to quit.\n")

    while True:
        # Step C: grab a frame.
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Step D: shrink the frame for faster processing.
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        # face_recognition expects RGB, OpenCV gives BGR — convert it.
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Step E: find every face location + encoding in this frame.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Step F: for each face found, compare it to all known encodings.
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=MATCH_TOLERANCE
            )
            name = "Unknown"

            # Step G: among the matches, pick the closest one (smallest distance).
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            face_names.append(name)

        # Step H: draw boxes + labels back on the ORIGINAL (full-size) frame.
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates back up since we shrank the frame earlier.
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

        cv2.imshow("Face Recognition", frame)

        # Step I: ESC key (27) exits.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Step J: cleanup.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()