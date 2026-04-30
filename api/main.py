from fastapi import FastAPI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from api.infer import infer_cnn, infer_resnet, infer_efficientnet

app = FastAPI()

class RequestData(BaseModel):
    img_data: str
    model_name: str  


@app.post("/predict")
def predict(data: RequestData):
    

    image_bytes = base64.b64decode(data.img_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")


    if data.model_name == "cnn":
       pred = infer_cnn(image)
    elif data.model_name == "resnet":
       pred = infer_resnet(image)
    elif data.model_name == "efficientnet":
       pred = infer_efficientnet(image)
    else:
       return {"error": "Invalid model name"}

    if hasattr(pred, "item"):
       pred = pred.item()

    return {
       "model": data.model_name,
       "prediction": pred
}