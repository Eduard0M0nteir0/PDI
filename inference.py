from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import numpy as np 
import joblib
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = 'C:\\Users\\DELL\\Downloads\\Apollo\\PDI\\aquarium_cropped\\test\\jellyfish\\3424.jpg'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(img_path)
image = image.convert('RGB')
image = transform(image)

image = image.unsqueeze(0)
image = image.to(device)

model = models.resnet50(pretrained=True)

model = nn.Sequential(*list(model.children())[:-1])
model.eval()

model.to(device)

outputs = model(image)
outputs = outputs.view(outputs.size(0), -1)
outputs = outputs.cpu().detach().numpy()
model_dt = joblib.load(os.path.join('models', "SVM.pkl"))
pred = model_dt.predict(outputs)
proba = model_dt.predict_proba(outputs)

print(pred)
print(proba)