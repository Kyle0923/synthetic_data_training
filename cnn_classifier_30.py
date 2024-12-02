import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Speed optimization for consistent input sizes

from torchvision.models import resnet18, ResNet18_Weights


# Define CNN model with reduced memory usage
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Using a pre-trained ResNet18 model and fine-tuning it
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, 2
        )  # 2 classes: colon_aca and colon_n

    def forward(self, x):
        return self.resnet(x)


# Custom dataset to load images from folders
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_images=None, seed=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_images = num_images
        self.img_paths = []
        self.labels = []

        if seed is not None:
            random.seed(seed)

        for label, subfolder in enumerate(["colon_aca", "colon_n"]):
            folder_path = os.path.join(root_dir, subfolder)
            img_files = os.listdir(folder_path)[:num_images]

            self.img_paths.extend([os.path.join(folder_path, img) for img in img_files])
            self.labels.extend([label] * len(img_files))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        print(f"Draw {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# Define transformations to normalize images
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to load data
def load_training_data(
    original_data_path,
    synthetic_data_path,
    num_original,
    num_synthetic,
    batch_size=1,
    seed=1,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_dataset = torch.utils.data.ConcatDataset(
        [
            CustomDataset(
                os.path.join(original_data_path, "train"),
                transform,
                num_original // 2,
                seed=seed,
            ),
            CustomDataset(
                os.path.join(synthetic_data_path),
                transform,
                num_synthetic // 2,
                seed=seed,
            ),
        ]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    print(f"Number of training images: {len(train_dataset)}")
    return train_loader


def load_test_data(original_data_path, batch_size=1):
    test_dataset = CustomDataset(os.path.join(original_data_path, "test"), transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return test_loader


# Training function with gradient accumulation and mixed precision
def train(
    model, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4
):
    model.train()
    running_loss = 0.0
    count = 0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    ):
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1} - Loss: {running_loss / count}")


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

    return accuracy, f1, auc


# Main function
def main():
    original_data_path = "./colon_image_sets_50"
    synthetic_data_path = "./synthetic_data"
    test_data_path = "./lung_colon_image_set/colon_image_sets"

    # Specify how many images from each dataset
    num_original1 = 30
    num_original2 = 50
    num_synthetic = 0

    test_loader = load_test_data(test_data_path)

    for num_orignal in range(num_original1, num_original2 + 1, 4):
        train_loader = load_training_data(
            original_data_path, synthetic_data_path, num_orignal, num_synthetic
        )

        model = CNNClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        accuracies = list()
        f1s = list()
        aucs = list()

        num_epochs = 5
        for epoch in range(num_epochs):
            train(model, train_loader, criterion, optimizer, scaler, epoch)
            accuracy, f1, auc = evaluate(model, test_loader)
            accuracies.append(accuracy)
            f1s.append(f1s)
            aucs.append(aucs)
            print(f"accuraries:\n{accuracies}\nf1s:\n{f1s}\naucs:{aucs}")


if __name__ == "__main__":
    main()
