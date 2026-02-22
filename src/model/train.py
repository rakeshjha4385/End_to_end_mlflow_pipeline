import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
# from model import SimpleCNN
from src.model.model import get_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    # ----------------------------
    # DATA AUGMENTATION (TRAIN)
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ----------------------------
    # VALIDATION / TEST TRANSFORM
    # ----------------------------
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_data = datasets.ImageFolder("data/split/train", transform=train_transform)
    val_data = datasets.ImageFolder("data/split/val", transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # model = SimpleCNN().to(DEVICE)
    model = get_model().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_experiment("CatsVsDogs")

    with mlflow.start_run():

        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 5)

        for epoch in range(5):

            model.train()
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # ----------------------------
        # VALIDATION
        # ----------------------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = (outputs > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)

        # ----------------------------
        # CONFUSION MATRIX
        # ----------------------------

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)


        print("Validation Accuracy:", accuracy)

        mlflow.log_metric("val_accuracy", accuracy)

        torch.save(model.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")

        print("Training Completed Successfully")


if __name__ == "__main__":
    train()
