from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import time
import joblib
import pandas as pd
import httpx

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


# Home page

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


# Health check

@app.get("/health")
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


# Weather details

CACHE_SECONDS = 1800  # reuse results for 30 minutes
_weather_cache = {}   # {city_key: (timestamp, result)}


async def fetch_json(client: httpx.AsyncClient, url: str, params: dict):
    """GET with retries when the API answers 429 (too many requests)."""
    for attempt in range(3):
        response = await client.get(url, params=params)
        if response.status_code == 429:
            await asyncio.sleep(2 ** attempt)  # wait 1s, 2s, 4s
            continue
        response.raise_for_status()
        return response.json()

    raise HTTPException(
        status_code=503,
        detail="Weather service is busy. Please try again in a minute."
    )


@app.get("/weather")
async def get_weather(city: str):

    cache_key = city.strip().lower()
    now = time.time()

    # Serve fresh cached result if we have one
    cached = _weather_cache.get(cache_key)
    if cached and now - cached[0] < CACHE_SECONDS:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=15) as client:

            # Find the location in India

            geo_data = await fetch_json(
                client,
                "https://geocoding-api.open-meteo.com/v1/search",
                {
                    "name": city,
                    "count": 10,
                    "language": "en",
                    "format": "json",
                    "countryCode": "IN"
                }
            )

            if not geo_data.get("results"):
                raise HTTPException(
                    status_code=404,
                    detail="Location not found in India. Please enter a valid Indian city."
                )

            # Filter only Indian locations

            indian_locations = [
                item for item in geo_data["results"]
                if item.get("country_code") == "IN"
            ]

            if not indian_locations:
                raise HTTPException(
                    status_code=404,
                    detail="No matching Indian location found."
                )

            # Prefer exact location name match

            location = next(
                (
                    item for item in indian_locations
                    if item.get("name", "").lower() == city.strip().lower()
                ),
                indian_locations[0]
            )

            # Get weather information

            weather_data = await fetch_json(
                client,
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,precipitation",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "forecast_days": 7,
                    "timezone": "auto"
                }
            )

            current = weather_data["current"]
            daily = weather_data["daily"]

            result = {
                "city": location["name"],
                "state": location.get("admin1", ""),
                "country": location.get("country", ""),
                "temperature": current["temperature_2m"],
                "humidity": current["relative_humidity_2m"],
                "current_precipitation": current["precipitation"],
                "date": current["time"],
                "forecast": {
                    "dates": daily["time"],
                    "max_temperature": daily["temperature_2m_max"],
                    "min_temperature": daily["temperature_2m_min"],
                    "rainfall": daily["precipitation_sum"]
                }
            }

            _weather_cache[cache_key] = (now, result)
            return result

    except HTTPException:
        # If the API is busy but we have an older result, serve that instead
        if cached:
            return cached[1]
        raise

    except httpx.HTTPError as e:
        if cached:
            return cached[1]
        raise HTTPException(
            status_code=502,
            detail=f"Weather API error: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Weather error: {str(e)}"
        )