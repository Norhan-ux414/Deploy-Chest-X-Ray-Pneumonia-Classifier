# Chest X-ray Pneumonia Classification

This project focuses on detecting pneumonia from chest X-ray images using deep learning.

I trained and evaluated three different models and built a simple FastAPI endpoint to make predictions using the trained models.

## Models

The project includes three models:

- Simple CNN
- ResNet18
- EfficientNet-B0

## Dataset

The dataset contains chest X-ray images divided into two classes:

- NORMAL
- PNEUMONIA

The data is split into training, validation, and testing sets.

Images are resized to 224 × 224 pixels. Grayscale images are used for the CNN and ResNet18 models, while EfficientNet-B0 uses three-channel grayscale images with ImageNet normalization.

## Model Evaluation

The models were evaluated using accuracy, precision, recall, and F1-score.

The reported test accuracy results were:

| Model | Test Accuracy |
|---|---:|
| Simple CNN | 81.57% |
| ResNet18 | 79.65% |
| EfficientNet-B0 | 87.50% |

These are experimental results from the current implementation. The evaluation settings should be standardized before making a final comparison.

## Project Structure

```text
xray_pneumonia_classifier/
│
├── api/
│   ├── main.py
│   ├── infer.py
│   └── image.py
│
├── data/
│   ├── chest_xray/
│   ├── processed/
│   └── data_map.csv
│
├── models/
│   ├── simple_cnn_best.pth
│   ├── resnet18_final.pth
│   └── efficientnet_best.pth
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

*Note: The structure above is a reference. Update folder and model filenames to match the actual files in the repository.*

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_LINK
cd xray_pneumonia_classifier
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the API

From the project root directory, run:

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The API provides a `/predict` endpoint that accepts a Base64-encoded image and a model name.

Supported model names:

- `cnn`
- `resnet`
- `efficientnet`

## Technologies Used

- Python
- PyTorch
- Torchvision
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Pillow

## Note

This project is for educational and experimental purposes only. It is not intended to be used as a medical diagnostic tool.
