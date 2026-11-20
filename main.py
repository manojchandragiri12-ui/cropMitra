from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI(
    title="Crop Suitability Prediction",
    description="ML Based Crop Recommendation System"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

model = joblib.load(BASE_DIR / "crop_model.pkl")


# Input validation

class CropInput(BaseModel):
    N: float = Field(ge=0, le=140)
    P: float = Field(ge=5, le=145)
    K: float = Field(ge=5, le=205)
    temperature: float = Field(ge=8.825675, le=43.675493)
    humidity: float = Field(ge=14.258040, le=99.981876)
    ph: float = Field(ge=3.504752, le=9.935091)
    rainfall: float = Field(ge=20.211267, le=298.560117)


# Home page (hidden from /docs)

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")


# Health check (hidden from /docs)

@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "Backend is running",
        "model": "Crop prediction model loaded"
    }


# Crop prediction

@app.post("/predict")
def predict_crop(data: CropInput):
    try:
        input_data = pd.DataFrame([{
            "N": data.N,
            "P": data.P,
            "K": data.K,
            "temperature": data.temperature,
            "humidity": data.humidity,
            "ph": data.ph,
            "rainfall": data.rainfall
        }])

        prediction = model.predict(input_data)

        return {
            "success": True,
            "predicted_crop": str(prediction[0])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Crop prediction failed: {str(e)}"
        )