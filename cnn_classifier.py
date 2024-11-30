import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset
import glob


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label, subfolder in enumerate(["colon_aca", "colon_n"]):
            class_path = os.path.join(self.data_dir, subfolder)
            class_images = glob.glob(os.path.join(class_path, "*.jpg"))
            for image_path in class_images:
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 96 * 96, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 96 * 96)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(train_loader, model, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                loss=running_loss / (total + 1e-5), accuracy=correct / total
            )

        print(
            f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}"
        )


def evaluate_model(test_loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_preds)

    return accuracy, f1, auc


def get_data_loaders(
    original_train_dir,
    synthetic_train_dir,
    batch_size,
    image_size,
    num_original,
    num_synthetic,
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    original_dataset = CustomImageDataset(
        data_dir=original_train_dir, transform=transform
    )
    synthetic_dataset = CustomImageDataset(
        data_dir=synthetic_train_dir, transform=transform
    )

    # Use only the specified number of images from each dataset
    original_subset = Subset(original_dataset, range(num_original))
    synthetic_subset = Subset(synthetic_dataset, range(num_synthetic))

    # Combine datasets
    full_train_dataset = original_subset + synthetic_subset

    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def main():
    original_train_dir = "./lung_colon_image_set/colon_image_sets/train"
    synthetic_train_dir = "synthetic_data"
    test_dir = "./lung_colon_image_set/colon_image_sets/test"

    # Hyperparameters
    image_size = 768
    batch_size = 32
    num_epochs = 10
    num_original = 2000  # Specify how many images from the original dataset
    num_synthetic = 0  # Specify how many images from the synthetic dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_data_loaders(
        original_train_dir,
        synthetic_train_dir,
        batch_size,
        image_size,
        num_original,
        num_synthetic,
    )

    # Train the model
    train_model(train_loader, model, criterion, optimizer, device, num_epochs)

    # Evaluate on test set
    test_dataset = CustomImageDataset(
        data_dir=test_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    accuracy, f1, auc = evaluate_model(test_loader, model, device)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
