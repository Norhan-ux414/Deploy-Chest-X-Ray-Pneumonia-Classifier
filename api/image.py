import base64

image_path = r"C:\Users\Nourhan Yehia\Desktop\Jupyter\xray_pneumonia_classifier\data\processed\test\PNEUMONIA\person1_virus_7.jpeg"

with open(image_path, "rb") as f:
    img_data = base64.b64encode(f.read()).decode("utf-8")

print(img_data)