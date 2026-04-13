from datasets import load_dataset
import numpy as np
from torchvision import transforms, models
import torch
from torch import tensor
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

# Cargar dataset
dataset = load_dataset("uoft-cs/cifar10")

# Dividir dataset
train_val = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Pesos
weights = tensor([1.0,1.0,1.2,1.5,1.1,1.4,1.0,1.2,1.0,1.1])
criterion = nn.CrossEntropyLoss(weight=weights)

# Transforms (para ResNet)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

print(dataset["train"].features["label"].names)

# USAR with_transform (NO map)
def transform_example(example, transform):
    images = example["img"]
    
    processed_images = []
    
    for img in images:
        img = np.array(img, dtype=np.uint8)
        
        # FIX CLAVE: quitar dimensiones extra
        if img.ndim == 4:
            img = img.squeeze()
        
        img = Image.fromarray(img)
        img = transform(img)
        processed_images.append(img)
    
    example["img"] = processed_images
    return example

train_dataset = train_dataset.with_transform(lambda x: transform_example(x, train_transforms))
val_dataset = val_dataset.with_transform(lambda x: transform_example(x, test_transforms))
test_dataset = test_dataset.with_transform(lambda x: transform_example(x, test_transforms))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

def train(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

# Entrenamiento CNN
best_val = 0

for epoch in range(10):  # más épocas
    loss = train(model, train_loader)
    val_acc = evaluate(model, val_loader)

    scheduler.step()

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_cnn.pth")

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")

# Cargar mejor modelo
model.load_state_dict(torch.load("best_cnn.pth"))

# Transfer Learning
model_tl = models.resnet18(weights="IMAGENET1K_V1")

for param in model_tl.parameters():
    param.requires_grad = False

# 🔥 Descongelar últimas capas (MEJORA IMPORTANTE)
for param in model_tl.layer4.parameters():
    param.requires_grad = True

model_tl.fc = nn.Linear(model_tl.fc.in_features, 10)
model_tl = model_tl.to(device)

optimizer_tl = torch.optim.Adam(model_tl.fc.parameters(), lr=0.001)

def train_tl(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        optimizer_tl.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer_tl.step()
        total_loss += loss.item()

    return total_loss / len(loader)

for epoch in range(5):
    loss = train_tl(model_tl, train_loader)
    val_acc = evaluate(model_tl, val_loader)
    print(f"[TL] Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")

test_acc_tl = evaluate(model_tl, test_loader)
print(f"Test Accuracy TL: {test_acc_tl:.4f}")

print("\nComparación:")
print(f"CNN: {test_acc:.4f}")
print(f"Transfer Learning: {test_acc_tl:.4f}")

print("\nRESULTADOS OPTIMIZADOS")
print(f"CNN Optimizada: {test_acc:.4f}")
print(f"Transfer Learning Optimizado: {test_acc_tl:.4f}")