import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# (3 channels)
transform_rgb = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# (1 channel)
transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["NORMAL", "PNEUMONIA"]



#  ResNet
import torch.nn as nn
from torchvision import models

resnet = models.resnet18(weights=None)  

resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

resnet.fc = nn.Linear(resnet.fc.in_features, 2)

resnet.load_state_dict(torch.load("models/resnet18_finall.pth", map_location=device))
resnet.to(device)
resnet.eval()

#  EfficientNet
efficientnet = models.efficientnet_b0(pretrained=False)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
efficientnet.load_state_dict(torch.load("models/efficientnet_final.pth", map_location=device))
efficientnet.to(device)
efficientnet.eval()

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 🔥 لازم موجود
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cnn = SimpleCNN().to(device)
cnn.load_state_dict(torch.load("models/model_01_simple_cnn_best.pth", map_location=device))
cnn.eval()

# =========================
#  INFERENCE FUNCTIONS
# =========================

def infer_resnet(img: Image.Image):
    img = transform_gray(img).unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = resnet(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)

    return classes[pred.item()]


def infer_efficientnet(img: Image.Image):
    img = transform_rgb(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = efficientnet(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)

    return classes[pred.item()]


def infer_cnn(img: Image.Image):
    img = transform_gray(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)

    return classes[pred.item()]

# main

if __name__ == "__main__":
    from PIL import Image

    img_path = r"C:\Users\Nourhan Yehia\Desktop\Jupyter\xray_pneumonia_classifier\data\processed\test\PNEUMONIA\person1_virus_7.jpeg"

    img = Image.open(img_path)

    print("ResNet:", infer_resnet(img))
    print("EfficientNet:", infer_efficientnet(img))
    print("CNN:", infer_cnn(img))