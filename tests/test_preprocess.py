from src.data.preprocess import preprocess_image
from PIL import Image
import os

def test_preprocess(tmp_path):
    input_path = "tests/sample.jpg"
    output_path = tmp_path / "out.jpg"

    preprocess_image(input_path, output_path)

    assert os.path.exists(output_path)
