import os
from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def preprocess_image(input_path: str, output_path: str):
    """Resize a single image to 224x224 RGB."""
    img = Image.open(input_path).convert("RGB")
    img = img.resize((224, 224))
    img.save(output_path)


def preprocess_folder(input_dir: str, output_dir: str):
    """Preprocess all valid images inside a directory."""
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):

        if file.startswith("."):
            continue

        if not file.lower().endswith(VALID_EXTENSIONS):
            continue

        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            preprocess_image(input_path, output_path)

        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {file}")
