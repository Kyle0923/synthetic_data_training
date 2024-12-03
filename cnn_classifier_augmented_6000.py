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
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torchvision.models import resnet18, ResNet18_Weights


# Define CNN model with reduced memory usage
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Using a pre-trained ResNet18 model and fine-tuning it
        self.resnet = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )  # Use weights instead of pretrained
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, 2
        )  # 2 classes: colon_aca and colon_n

    def forward(self, x):
        return self.resnet(x)


import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer: Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(
            3, 8, kernel_size=3, stride=1, padding=1
        )  # 3 input channels (RGB), 8 output channels

        # MaxPooling layer: reducing spatial dimensions (2x2 window)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer (reduce size by factor of 2)

        # Fully connected layers
        # After one convolution + pooling, the image size will be reduced to 384x384 (768 -> 384)
        self.fc1 = nn.Linear(8 * 384 * 384, 256)  # 8 channels, 384x384 after pooling
        self.fc2 = nn.Linear(256, 2)  # 2 classes: colon_aca and colon_n

    def forward(self, x):
        # Pass through first convolutional block
        x = self.pool(torch.relu(self.conv1(x)))  # Apply ReLU activation and MaxPooling

        # Flatten the tensor for feeding into fully connected layers
        x = x.view(
            -1, 8 * 384 * 384
        )  # Flatten the tensor (batch_size, channels * height * width)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Final layer for 2 output classes

        return x


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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# Define transformations to resize and normalize images
transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),  # Reduce size to minimize memory usage
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
    batch_size=4,
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


def load_test_data(
    original_data_path,
    batch_size=4,
    seed=1,
):

    test_dataset = CustomDataset(os.path.join(original_data_path, "test"), transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return test_loader


# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    count = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

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

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1-score: {f1:.4f}")
    # print(f"AUC: {auc:.4f}")

    return accuracy, f1, auc


# Main function
def main():
    original_data_path = "./lung_colon_image_set/colon_image_sets"
    synthetic_data_path = "./synthetic_data"

    # Specify how many images from each dataset
    num_original = 6000
    num_synthetic = 0
    epochs = 20

    test_loader = load_test_data(original_data_path)

    train_loader_6_0 = load_training_data(
        original_data_path, synthetic_data_path, num_original, num_synthetic
    )

    num_synthetic = 2000
    train_loader_6_2 = load_training_data(
        original_data_path, synthetic_data_path, num_original, num_synthetic
    )

    num_synthetic = 6000
    train_loader_6_6 = load_training_data(
        original_data_path, synthetic_data_path, num_original, num_synthetic
    )

    num_original = 2000
    train_loader_2_6 = load_training_data(
        original_data_path, synthetic_data_path, num_original, num_synthetic
    )

    train_loaders = [
        train_loader_6_0,
        train_loader_6_2,
        train_loader_6_6,
        train_loader_2_6,
    ]

    model_descriptions = [
        "6000 real images, 0 synthetic images",
        "6000 real images, 2000 synthetic images",
        "6000 real images, 6000 synthetic images",
        "2000 real images, 6000 synthetic images",
    ]

    for train_loader, title in zip(train_loaders, model_descriptions):
        print(f"Training model with {title}")
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        accuracies = list()
        f1s = list()
        aucs = list()
        num_epochs = epochs
        for epoch in range(1, num_epochs + 1):
            train(model, train_loader, criterion, optimizer, epoch)
            accuracy, f1, auc = evaluate(model, test_loader)
            accuracies.append(accuracy)
            f1s.append(f1)
            aucs.append(auc)
        print(f"With{title}\naccuracy:\n{accuracies}\nF1:\n{f1s}\nacus:\n{aucs}")


if __name__ == "__main__":
    main()
