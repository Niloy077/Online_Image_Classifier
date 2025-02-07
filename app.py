from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import json

app = Flask(__name__)

# Load Pretrained ResNet18 Model
# model = models.resnet18(pretrained=True)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load ImageNet Class Labels
LABELS = json.load(open("imagenet_classes.json"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = outputs.max(1)

    class_label = LABELS[predicted_class.item()]
    return jsonify({"prediction": class_label})

if __name__ == "__main__":
    app.run(debug=True)
