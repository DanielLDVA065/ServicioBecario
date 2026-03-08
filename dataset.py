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
