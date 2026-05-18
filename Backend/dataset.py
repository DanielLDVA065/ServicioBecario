from datasets import load_dataset
import numpy as np
from torchvision import transforms, models
import torch
from torch import tensor
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import os

# FASTAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

# CARGA DE DATOS
dataset = load_dataset("uoft-cs/cifar10")

# Dividir en train, validation y test
train_val = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Pesos para clases
weights = tensor([1.0,1.0,1.2,1.5,1.1,1.4,1.0,1.2,1.0,1.1])
criterion = nn.CrossEntropyLoss(weight=weights)

# TRANSFORMACIONES
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Transformaciones para test
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

print(dataset["train"].features["label"].names)

# TRANSFORMACIÓN DEL DATASET
# Convierte imágenes y aplica transformaciones

def transform_example(example, transform):

    images = example["img"]
    processed_images = []

    for img in images:

        img = np.array(img, dtype=np.uint8)

        # Quitar dimensiones extra
        if img.ndim == 4:
            img = img.squeeze()

        img = Image.fromarray(img)

        img = transform(img)

        processed_images.append(img)

    example["img"] = processed_images

    return example

# Aplicar transformaciones
train_dataset = train_dataset.with_transform(
    lambda x: transform_example(x, train_transforms)
)

val_dataset = val_dataset.with_transform(
    lambda x: transform_example(x, test_transforms)
)

test_dataset = test_dataset.with_transform(
    lambda x: transform_example(x, test_transforms)
)

# DataLoaders
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

        # Dropout
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

# Crear modelo
model = SimpleCNN().to(device)

# Cargar modelo guardado
model.load_state_dict(torch.load("Backend/cnn_model.pth", map_location=device))

# Modo evaluación
model.eval()

# Clases
classes = dataset["train"].features["label"].names

# FUNCIÓN DE PREDICCIÓN

def predict_image(image):

    image = image.convert("RGB")

    image = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]

# FASTAPI
app = FastAPI()

# Carpeta static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates HTML
templates = Jinja2Templates(directory="templates")

# Página principal
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# Endpoint predict
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image = Image.open(file.file)

        prediction = predict_image(image)

        return {
            "prediction": prediction
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# Ejecutar servidor
if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)