from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
from typing import Dict, List

app = FastAPI(title="Skin Lesion Classifier API")


class ModelService:
    """Model service for inference"""

    def __init__(self, model_path: str, config_path: str):
        self.model = self.load_model(model_path)
        self.config = self.load_config(config_path)
        self.transform = self.get_transform()
        self.class_names = self.load_class_names()

    async def predict(self, image_bytes: bytes) -> Dict:
        """Predict from image bytes"""
        # Preprocess
        image = Image.open(io.BytesIO(image_bytes))
        tensor = self.preprocess(image)

        # Predict
        with torch.no_grad():
            output = self.model(tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1)

        # Format response
        top_k = 3
        values, indices = probs.topk(top_k)

        predictions = []
        for i in range(top_k):
            predictions.append(
                {
                    "class": self.class_names[indices[0][i].item()],
                    "confidence": values[0][i].item(),
                }
            )

        return {"predictions": predictions, "gradcam": self.generate_gradcam(tensor)}


# Initialize service
model_service = ModelService("models/best_model.pth", "configs/config.yaml")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict skin lesion class"""
    try:
        contents = await file.read()
        results = await model_service.predict(contents)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
