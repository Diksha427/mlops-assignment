from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import logging
import time

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------
# Load trained model
# ---------------------------
model = load("model.joblib")

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="Heart Disease Prediction API")

# ---------------------------
# Monitoring variables
# ---------------------------
request_count = 0

# ---------------------------
# Input schema
# ---------------------------
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict(data: HeartInput):
    global request_count
    request_count += 1

    start_time = time.time()

    try:
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps,
            data.chol, data.fbs, data.restecg, data.thalach,
            data.exang, data.oldpeak, data.slope, data.ca,
            data.thal
        ]])

        prediction = model.predict(input_data)[0]

        latency = time.time() - start_time

        # Logging
        logging.info(
            f"Request #{request_count} | Input={data.dict()} | "
            f"Prediction={int(prediction)} | Latency={latency:.4f}s"
        )

        return {
            "prediction": int(prediction),
            "latency_seconds": round(latency, 4)
        }

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": "Prediction failed"}
