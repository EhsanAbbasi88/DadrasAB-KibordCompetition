import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ESP32 AI Server")


class SensorData(BaseModel):
    sensor1: float
    sensor2: float
    sensor3: float


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Server is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sensors")
def receive_sensors(data: SensorData):
    return {
        "received": True,
        "sensor1": data.sensor1,
        "sensor2": data.sensor2,
        "sensor3": data.sensor3
    }


@app.post("/analyze")
def analyze_sensors(data: SensorData):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenRouter API key is not configured"
        )

    payload = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a sensor-data analysis assistant. "
                    "Analyze the supplied numeric readings carefully. "
                    "Explain possible interpretations and uncertainty. "
                    "Do not invent sensor units or claim a diagnosis."
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze these sensor readings:\n"
                    + data.model_dump_json()
                )
            }
        ],
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        response.raise_for_status()
        result = response.json()

        print("OpenRouter raw response:", result, flush=True)

        answer = result["choices"][0]["message"]["content"]
        return {
            "received": True,
            "analysis": answer,
            "model": result.get("model")
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not get a response from OpenRouter"
        )
    except (KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from OpenRouter"
        )
