import os
from PIL import Image, UnidentifiedImageError

INPUT_DIR = "data/raw/training_set/training_set"
OUTPUT_DIR = "data/processed"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def preprocess():

    for label in ["cats", "dogs"]:

        input_path = os.path.join(INPUT_DIR, label)
        output_path = os.path.join(OUTPUT_DIR, label)

        os.makedirs(output_path, exist_ok=True)

        for file in os.listdir(input_path):

            # Skip hidden files
            if file.startswith("."):
                continue

            # Skip non-image files
            if not file.lower().endswith(VALID_EXTENSIONS):
                continue

            try:
                img_path = os.path.join(input_path, file)
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                img.save(os.path.join(output_path, file))

            except UnidentifiedImageError:
                print(f"Skipping corrupted image: {file}")

    print("Preprocessing Completed Successfully")


if __name__ == "__main__":
    preprocess()
