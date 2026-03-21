from datasets import load_dataset
import numpy as np
from torchvision import transforms
import torch
from torch import tensor
import torch.nn as nn

dataset = load_dataset("uoft-cs/cifar10")

# Normalización de las imágenes
def normalize(example):
    image = np.array(example["img"]) / 255.0
    example["img"] = image
    return example
dataset = dataset.map(normalize)

# Dividir el conjunto de entrenamiento en entrenamiento y validación
train_val = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Calcular los pesos para cada clase en función de su frecuencia
weights = tensor([1.0,1.0,1.2,1.5,1.1,1.4,1.0,1.2,1.0,1.1])
criterion = nn.CrossEntropyLoss(weight=weights)

# Definir las transformaciones para el entrenamiento y la prueba
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Para el conjunto de prueba, solo normalizamos sin aumentos de datos
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


#Comprobacion
print(dataset["train"].features["label"].names) # Imprime las clases del dataset

from torch.utils.data import DataLoader

# Convertir dataset a formato PyTorch
def transform_dataset(example, transform):
    image = example["img"]
    image = transform(image)
    example["img"] = image
    return example

train_dataset = train_dataset.map(lambda x: transform_dataset(x, train_transforms))
val_dataset = val_dataset.map(lambda x: transform_dataset(x, test_transforms))
test_dataset = test_dataset.map(lambda x: transform_dataset(x, test_transforms))

train_dataset.set_format(type="torch", columns=["img", "label"])
val_dataset.set_format(type="torch", columns=["img", "label"])
test_dataset.set_format(type="torch", columns=["img", "label"])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(self.relu(self.conv2(x)))  # 64x8x8
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

epochs = 5

for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_acc = evaluate(model, val_loader)
    
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")