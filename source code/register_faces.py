


import cv2
import os

# Folder where all registered face images will live.
KNOWN_FACES_DIR = "known_faces"


def get_next_filename(person_dir):
    """Look inside person_dir and figure out the next free filename,
    so repeated runs don't overwrite previous photos (img_1.jpg, img_2.jpg, ...)."""
    existing = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    return os.path.join(person_dir, f"img_{len(existing) + 1}.jpg")


def main():
    # Step A: make sure the base known_faces/ folder exists.
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    # Step B: ask for the person's name.
    name = input("Enter the name of the person to register: ").strip()
    if not name:
        print("Name cannot be empty. Exiting.")
        return

    # Step C: create a subfolder for this person, e.g. known_faces/John/
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Step D: load OpenCV's built-in face detector (Haar Cascade).
    # This is ONLY used here to draw a box so you can see your face is detected.
    # The actual recognition encoding happens later in encode_faces.py.
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Step E: open the webcam. 0 = default camera.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera connection/permissions.")
        return

    print("\nWebcam started.")
    print("  -> Press SPACE to capture a photo of your face.")
    print("  -> Press ESC to quit.\n")

    photos_taken = 0

    while True:
        # Step F: read one frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Step G: convert to grayscale (Haar Cascade needs grayscale input).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step H: detect faces in the current frame.
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # Step I: draw a rectangle around every detected face so the user
        # gets visual feedback.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Step J: show instructions on-screen too.
        cv2.putText(
            frame,
            f"Photos taken: {photos_taken}  |  SPACE=capture  ESC=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Register Face - " + name, frame)

        key = cv2.waitKey(1) & 0xFF

        # Step K: ESC key (27) exits the loop.
        if key == 27:
            break

        # Step L: SPACE key (32) saves the current frame if a face was found.
        if key == 32:
            if len(faces) == 0:
                print("No face detected, try again.")
                continue

            filename = get_next_filename(person_dir)
            cv2.imwrite(filename, frame)
            photos_taken += 1
            print(f"Saved: {filename}")

    # Step M: cleanup — always release the camera and close windows.
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. {photos_taken} photo(s) saved to {person_dir}/")
    print("Next step: run  python encode_faces.py  to generate encodings.")


if __name__ == "__main__":
    main()