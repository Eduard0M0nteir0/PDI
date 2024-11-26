import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import joblib

def extract_features(data_loader, model, device):
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('aquarium_cropped/train', transform=transform)
val_dataset = datasets.ImageFolder('aquarium_cropped/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=True)

# Drop FC layer 
model = nn.Sequential(*list(model.children())[:-1])

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_features, train_labels = extract_features(train_loader, model, device)
val_features, val_labels = extract_features(val_loader, model, device)

joblib.dump((train_features, train_labels), 'train_features.pkl')
joblib.dump((val_features, val_labels), 'val_features.pkl')