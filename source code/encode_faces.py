import os
import pickle
import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"


def main():
    # Step A: these two lists will grow together — index i in
    # known_encodings matches index i in known_names.
    known_encodings = []
    known_names = []

    # Step B: make sure the known_faces folder actually exists first.
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"'{KNOWN_FACES_DIR}/' not found. Run register_faces.py first.")
        return

    # Step C: loop over every person-folder inside known_faces/.
    person_names = sorted(os.listdir(KNOWN_FACES_DIR))
    if not person_names:
        print("No registered people found. Run register_faces.py first.")
        return

    for name in person_names:
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue  # skip stray files

        print(f"Processing images for: {name}")

        # Step D: loop over every image of this person.
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(person_dir, image_name)

            # Step E: load the image into memory.
            image = face_recognition.load_image_file(image_path)

            # Step F: find the pixel-location of each face in the image.
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 0:
                print(f"  No face found in {image_name}, skipping.")
                continue

            # Step G: convert the (first) detected face into a 128-d encoding.
            encodings = face_recognition.face_encodings(image, face_locations)
            face_encoding = encodings[0]

            # Step H: store the encoding + the person's name together.
            known_encodings.append(face_encoding)
            known_names.append(name)
            print(f"  Encoded {image_name}")

    # Step I: save everything to disk as a single pickle file.
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\nDone. {len(known_encodings)} face encoding(s) saved to {ENCODINGS_FILE}")
    print("Next step: run  python face_recognizer.py  to test live recognition.")


if __name__ == "__main__":
    main()