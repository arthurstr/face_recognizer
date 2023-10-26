from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import argparse

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

class FaceRecognizer:
    def __init__(self, encodings_location=Path(DEFAULT_ENCODINGS_PATH)):
        self.encodings_location = encodings_location
        self.model = "hog"

    def encode_known_faces(self, train_dir):
        names = []
        encodings = []

        for filepath in train_dir.glob("*/*"):
            if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                name = filepath.parent.name
                image = face_recognition.load_image_file(filepath)

                face_locations = face_recognition.face_locations(image, model=self.model)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for encoding in face_encodings:
                    names.append(name)
                    encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
        with self.encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)

    def recognize_faces(self, image_location):
        with self.encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        input_image = face_recognition.load_image_file(image_location)
        input_face_locations = face_recognition.face_locations(input_image, model=self.model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name = self._recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            self._display_face(draw, bounding_box, name)

        del draw
        pillow_image.show()

    def _recognize_face(self, unknown_encoding, loaded_encodings):
        boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
        votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
        if votes:
            return votes.most_common(1)[0][0]

    def _display_face(self, draw, bounding_box, name):
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
        draw.rectangle(
            ((text_left, text_top), (text_right, text_bottom)),
            fill=BOUNDING_BOX_COLOR,
            outline=BOUNDING_BOX_COLOR,
        )
        draw.text(
            (text_left, text_top),
            name,
            fill=TEXT_COLOR,
        )

# Create a new agent for training
class Trainer:
    def __init__(self, train_dir):
        self.train_dir = train_dir

    def train(self):
        face_recognizer = FaceRecognizer()
        face_recognizer.encode_known_faces(self.train_dir)

# Create a new agent for validation
class Validator:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def validate(self):
        for filepath in Path(self.directory_path).rglob("*"):
            if filepath.is_file():
                face_recognizer = FaceRecognizer()
                face_recognizer.recognize_faces(filepath)

# Create a new agent for testing
class Tester:
    def __init__(self, image_location):
        self.image_location = image_location

    def test(self):
        face_recognizer = FaceRecognizer()
        face_recognizer.recognize_faces(self.image_location)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Система распознавания лиц")
    parser.add_argument("--train", action="store_true", help="Обучить модель")
    parser.add_argument("--validate", action="store_true", help="Проверить модель")
    parser.add_argument("--test", action="store_true", help="Протестировать модель на изображении с неизвестным лицом")
    parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Модель для обучения: hog (на CPU), cnn (на GPU)")
    parser.add_argument("-f", action="store", help="Путь к изображению с неизвестным лицом")
    args = parser.parse_args()

    if args.train:
        train_dir = Path("training")  # Замените на путь к папке с изображениями для обучения
        trainer = Trainer(train_dir)
        trainer.train()

    if args.validate:
        validator = Validator("validation")
        validator.validate()

    if args.test:
        if args.f:
            tester = Tester(args.f)
            tester.test()
        else:
            print("Пожалуйста, укажите путь к изображению с неизвестным лицом, используя опцию -f.")

