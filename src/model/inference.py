import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def predict_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    return float(output.item())
