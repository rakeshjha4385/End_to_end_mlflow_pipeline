from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
from torchvision import transforms
from src.model.model import get_model
# from src.model.model import SimpleCNN
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# model = SimpleCNN()
model = get_model()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

request_count = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):

    global request_count
    request_count += 1
    logging.info(f"Prediction request #{request_count}")

    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    probability = float(output.item())
    label = "dog" if probability > 0.5 else "cat"

    return {
        "probability": probability,
        "label": label
    }
