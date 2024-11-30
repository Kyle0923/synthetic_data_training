import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define CNN model with reduced memory usage
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Using a pre-trained ResNet18 model and fine-tuning it
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, 2
        )  # 2 classes: colon_aca and colon_n

    def forward(self, x):
        return self.resnet(x)


# Custom dataset to load images from folders
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_images=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_images = num_images
        self.img_paths = []
        self.labels = []

        for label, subfolder in enumerate(["colon_aca", "colon_n"]):
            folder_path = os.path.join(root_dir, subfolder)
            img_files = (
                os.listdir(folder_path)[:num_images]
                if num_images
                else os.listdir(folder_path)
            )
            self.img_paths.extend([os.path.join(folder_path, img) for img in img_files])
            self.labels.extend([label] * len(img_files))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# Define transformations to resize and normalize images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Reduce size to minimize memory usage
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to load data
def load_data(original_data_path, synthetic_data_path, num_original, num_synthetic):
    train_dataset = torch.utils.data.ConcatDataset(
        [
            CustomDataset(
                os.path.join(original_data_path, "train"), transform, num_original // 2
            ),
            CustomDataset(
                os.path.join(synthetic_data_path), transform, num_synthetic // 2
            ),
        ]
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    test_dataset = CustomDataset(os.path.join(original_data_path, "test"), transform)
    print(
        f"Number of test images: {len(test_dataset)}"
    )  # Print the number of test images
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    return train_loader, test_loader


# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader)}")


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")


# Main function
def main():
    original_data_path = "./lung_colon_image_set/colon_image_sets"
    synthetic_data_path = "./synthetic_data"

    # Specify how many images from each dataset
    num_original = 2000
    num_synthetic = 1000

    train_loader, test_loader = load_data(
        original_data_path, synthetic_data_path, num_original, num_synthetic
    )

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)

    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
