from pathlib import Path
import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)

            face_locations = face_recognition.face_locations(image, model=model) #местоположение лиц
            face_encodings = face_recognition.face_encodings(image, face_locations) #кодировка только лица

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

    name_encodings = { "names" : names, "encodings" : encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()