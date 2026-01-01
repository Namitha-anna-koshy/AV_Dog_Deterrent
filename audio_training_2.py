#Training to avoid overfitting and flase postive by correcting the threshold values
# ===================== SETUP =====================
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/Audio_training"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ===================== DATASET CLASS =====================
class DogSoundDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # If the dataset images are .npy (Mel spectrogram arrays)
        mel = np.load(file_path)      # shape e.g. (128, 256)
        mel = np.stack([mel, mel, mel], axis=0)  # convert to 3-channel
        mel = torch.tensor(mel, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label


# ===================== LOAD FILES =====================
aggressive_files = [os.path.join(DATA_DIR, "aggressive", f)
                    for f in os.listdir(os.path.join(DATA_DIR, "aggressive")) if f.endswith(".npy")]

non_aggressive_files = [os.path.join(DATA_DIR, "non_aggressive", f)
                        for f in os.listdir(os.path.join(DATA_DIR, "non_aggressive")) if f.endswith(".npy")]

file_paths = aggressive_files + non_aggressive_files
labels = [1] * len(aggressive_files) + [0] * len(non_aggressive_files)

train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, shuffle=True, random_state=42
)

train_dataset = DogSoundDataset(train_files, train_labels)
val_dataset = DogSoundDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# ===================== MODEL (RESNET18) =====================
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary output

model = model.to(DEVICE)


# ===================== TRAINING SETUP =====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
EPOCHS = 25


# ===================== TRAINING LOOP =====================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0

    for mel, label in train_loader:
        mel, label = mel.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (output.argmax(1) == label).sum().item()

    acc = correct / len(train_dataset)

    # --------- Validation ----------
    model.eval()
    val_correct = 0
    val_loss_total = 0

    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            out = model(mel)
            val_loss_total += criterion(out, label).item()
            val_correct += (out.argmax(1) == label).sum().item()

    val_acc = val_correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {acc:.3f} | Val Acc: {val_acc:.3f}")


# ===================== SAVE MODEL =====================
torch.save(model.state_dict(), "/content/drive/MyDrive/aud_aggressive_classifier.pth")
print("Model saved successfully!")
