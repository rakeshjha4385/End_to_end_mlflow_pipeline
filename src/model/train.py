import torch
import torch.optim as optim
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("data/processed/train", transform=transform)
    val_data = datasets.ImageFolder("data/processed/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    model = SimpleCNN().to(DEVICE)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_experiment("CatsVsDogs")

    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        mlflow.log_param("epochs", 5)

        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")

        print("Training completed")

if __name__ == "__main__":
    train()
