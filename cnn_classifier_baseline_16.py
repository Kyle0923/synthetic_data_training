import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Speed optimization for consistent input sizes


# Define CNN model with optimized memory usage
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)


# Custom dataset to load images
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_images=None, random_pick=False):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for label, subfolder in enumerate(["colon_aca", "colon_n"]):
            folder_path = os.path.join(root_dir, subfolder)
            if random_pick:
                img_files = os.listdir(folder_path)  # Get all files in the folder
                img_files = random.sample(img_files, num_images)
            else:
                img_files = sorted(os.listdir(folder_path))[:num_images]
            # print(f"{len(img_files)} drawn")
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


# Define transformations
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to load training data
def load_training_data(
    original_data_path,
    synthetic_data_path,
    num_original,
    num_synthetic,
    batch_size=1,
    random_pick=False,
):
    torch.manual_seed(42)
    train_dataset = torch.utils.data.ConcatDataset(
        [
            CustomDataset(
                os.path.join(original_data_path, "train"),
                transform=transform,
                num_images=num_original // 2,
                random_pick=random_pick,
            ),
            CustomDataset(
                synthetic_data_path,
                transform=transform,
                num_images=num_synthetic // 2,
                random_pick=random_pick,
            ),
        ]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    loader_size = len(train_loader)
    print(f"{loader_size * batch_size} real data")
    return train_loader


# Function to load test data
def load_test_data(test_data_path, batch_size=1):
    test_dataset = CustomDataset(
        os.path.join(test_data_path, "test"), transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return test_loader


# Training function
def train(
    model, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4
):
    model.train()
    running_loss = 0.0
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

    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader)}")


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs)

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1-score: {f1:.4f}")
    # print(f"AUC: {auc:.4f}")
    return accuracy, f1, auc


# Main function
def main():
    # original_data_path_50 = "./lung_colon_image_set/colon_image_sets"
    original_data_path_all = "./lung_colon_image_set/colon_image_sets"
    synthetic_data_path = "./synthetic_data"
    test_data_path = "./lung_colon_image_set/colon_image_sets"

    num_original_start = 4
    num_original_end = 84
    num_synthetic = 0
    batch_size = 4
    epochs = 200

    test_loader = load_test_data(test_data_path, batch_size=batch_size)

    for num_original in range(num_original_start, num_original_end + 1, 8):
        train_loader_fixed = load_training_data(
            original_data_path_all,
            synthetic_data_path,
            num_original,
            num_synthetic,
            batch_size=batch_size,
            random_pick=False,
        )

        train_loader_random = load_training_data(
            original_data_path_all,
            synthetic_data_path,
            num_original,
            num_synthetic,
            batch_size=batch_size,
            random_pick=True,
        )

        model = CNNClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        accuracies = list()
        f1s = list()
        aucs = list()

        print("train with fixed first images")
        for epoch in range(1, epochs + 1):
            train(model, train_loader_fixed, criterion, optimizer, scaler, epoch)
            accuracy, f1, auc = evaluate(model, test_loader)
            accuracies.append(accuracy)
            f1s.append(f1)
            aucs.append(auc)
            if epoch % 40 == 0:
                print(f"Accuracy:\n{accuracies}\nF1:\n{f1s}\nAUC\n{aucs}")

        model = CNNClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        accuracies = list()
        f1s = list()
        aucs = list()

        print("train with random images")
        for epoch in range(epochs):
            train(model, train_loader_random, criterion, optimizer, scaler, epoch)
            accuracy, f1, auc = evaluate(model, test_loader)
            accuracies.append(accuracy)
            f1s.append(f1)
            aucs.append(auc)
        print(f"Accuracy:\n{accuracies}\nF1:\n{f1s}\nAUC\n{aucs}")


if __name__ == "__main__":
    main()
