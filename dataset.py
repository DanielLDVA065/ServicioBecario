import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import traceback

# FASTAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

# ------------------ CONFIGURACIÓN ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Transformaciones más ligeras para reducir memoria y tiempo de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reducido de 224 a 128 para ahorrar memoria
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------ MODELO ------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # Después de dos maxpool con imagen 128 -> 64 -> 32
        self.fc1 = nn.Linear(32 * 32 * 32, 64)  # 32*32*32 = 32768
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ------------------ CARGA DE DATOS ------------------
def get_data_loaders(batch_size=32):  # batch reducido para menor memoria
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    # Dividir train en train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader, test_loader

# ------------------ ENTRENAMIENTO ------------------
model_path = "Backend/cnn_model.pth"
os.makedirs("Backend", exist_ok=True)

if not os.path.exists(model_path):
    print("Modelo no encontrado. Entrenando uno nuevo (3 épocas)...")
    try:
        train_loader, val_loader, _ = get_data_loaders(batch_size=32)
        model = SmallCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(3):
            model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:  # cada 100 batches muestra progreso
                    print(f"  Batch {i+1}/{len(train_loader)}, loss: {running_loss/(i+1):.4f}")
            print(f"Época {epoch+1}/3 - Pérdida promedio: {running_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), model_path)
        print(f"Modelo guardado exitosamente en {model_path}")
        
        # Verificar que se guardó
        if os.path.exists(model_path):
            print(f"Archivo creado: {model_path} (tamaño: {os.path.getsize(model_path)} bytes)")
        else:
            print("ERROR: No se pudo guardar el archivo.")
            
    except Exception as e:
        print("ERROR durante el entrenamiento:")
        traceback.print_exc()
        # Crear un modelo dummy de respaldo para que la API funcione
        print("Creando modelo dummy de emergencia...")
        model = SmallCNN().to(device)
        torch.save(model.state_dict(), model_path)
        print(f"Modelo dummy guardado en {model_path}")
else:
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Modelo cargado desde disco")

model.eval()

# ------------------ PREDICCIÓN ------------------
def predict_image(image):
    image = image.convert("RGB")
    image = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return classes[pred.item()]

# ------------------ FASTAPI ------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        pred = predict_image(image)
        return {"prediction": pred}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)