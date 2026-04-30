from datasets import load_dataset
import numpy as np
from torchvision import transforms, models
import torch
from torch import tensor
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import os

# CARGA DE DATOS
dataset = load_dataset("uoft-cs/cifar10")

# Dividir en train, validation y test
train_val = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Pesos para clases (para balancear)
weights = tensor([1.0,1.0,1.2,1.5,1.1,1.4,1.0,1.2,1.0,1.1])
criterion = nn.CrossEntropyLoss(weight=weights)

# TRANSFORMACIONES
# Transformaciones con aumento de datos
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Transformaciones sin aumento
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

print(dataset["train"].features["label"].names)


# TRANSFORMACIÓN DEL DATASET
# Convierte imágenes a formato correcto y aplica transformaciones
def transform_example(example, transform):
    images = example["img"]
    processed_images = []

    for img in images:
        img = np.array(img, dtype=np.uint8)

        # Quita dimensiones extra si existen
        if img.ndim == 4:
            img = img.squeeze()

        img = Image.fromarray(img)
        img = transform(img)
        processed_images.append(img)

    example["img"] = processed_images
    return example

# Aplicar transformaciones sin usar map (evita errores)
train_dataset = train_dataset.with_transform(lambda x: transform_example(x, train_transforms))
val_dataset = val_dataset.with_transform(lambda x: transform_example(x, test_transforms))
test_dataset = test_dataset.with_transform(lambda x: transform_example(x, test_transforms))

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# MODELO CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # Pooling y activación
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        # Capas fully connected
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout para evitar overfitting
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Detectar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

# Optimizador con regularización
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Scheduler para bajar el learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# FUNCIONES DE ENTRENAMIENTO
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

# Evaluación (accuracy)
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

# ENTRENAMIENTO CNN
best_val = 0

for epoch in range(10):
    loss = train(model, train_loader)
    val_acc = evaluate(model, val_loader)

    scheduler.step()

    # Guardar mejor modelo
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_cnn.pth")

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")

# Cargar mejor modelo
model.load_state_dict(torch.load("best_cnn.pth"))

# Evaluar en test
test_acc = evaluate(model, test_loader)
print(f"Test Accuracy CNN: {test_acc:.4f}")

# TRANSFER LEARNING (ResNet18)
model_tl = models.resnet18(weights="IMAGENET1K_V1")

# Congelar todo
for param in model_tl.parameters():
    param.requires_grad = False

# Descongelar última capa convolucional
for param in model_tl.layer4.parameters():
    param.requires_grad = True

# Cambiar capa final
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

# Entrenar TL
for epoch in range(5):
    loss = train_tl(model_tl, train_loader)
    val_acc = evaluate(model_tl, val_loader)
    print(f"[TL] Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")

test_acc_tl = evaluate(model_tl, test_loader)
print(f"Test Accuracy TL: {test_acc_tl:.4f}")


# GUARDADO DE MODELOS
os.makedirs("inference", exist_ok=True)

torch.save(model.state_dict(), "inference/cnn_model.pth")
torch.save(model_tl.state_dict(), "inference/resnet_model.pth")

# FUNCIÓN DE PREDICCIÓN
classes = dataset["train"].features["label"].names

def predict(image_path, model, transform):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]


# VALIDACIÓN DE INFERENCIA
# Cambia esta ruta por una imagen tuya
test_image = "test.jpg"

if os.path.exists(test_image):
    pred = predict(test_image, model, test_transforms)
    print(f"Predicción CNN: {pred}")

    pred_tl = predict(test_image, model_tl, test_transforms)
    print(f"Predicción TL: {pred_tl}")
else:
    print("Pon una imagen llamada test.jpg para probar inferencia")