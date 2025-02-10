import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json

# Load ResNet18 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)

# Load ImageNet class labels
LABELS_PATH = "imagenet_classes.json"
with open(LABELS_PATH) as f:
    labels_map = json.load(f)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="ResNet18 Image Classifier", layout="centered")

st.title("🖼️ ResNet18 Image Classifier")
st.write("Upload an image to classify using a pre-trained ResNet18 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image and make prediction
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)

    predicted_label = labels_map[str(predicted.item())]

    st.success(f"### 🎯 Prediction: **{predicted_label}**")
