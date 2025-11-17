# --- Libararies ---
import pandas as pd
import joblib
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response



# --- Loading parameters from configuaration file ---
with open('integration_config.json', 'r') as f:
    config = json.load(f)

logistic_regression_path = config["ModelsPaths"]["logistic_regression_path"]
path_to_scaler = config["ModelsPaths"]["path_to_scaler"]

min_temperature = config["CharacteristicLimits"]["min_temperature"]
max_temperature = config["CharacteristicLimits"]["max_temperature"]
min_humidity = config["CharacteristicLimits"]["min_humidity"]
max_humidity = config["CharacteristicLimits"]["max_humidity"]
min_sound_volume = config["CharacteristicLimits"]["min_sound_volume"]
max_sound_volume = config["CharacteristicLimits"]["max_sound_volume"]

# --- Loading models ---
logistic_regression = joblib.load(logistic_regression_path)
scaler = joblib.load(path_to_scaler)

# ----- APP -----
app = FastAPI(title="Anomaly Detection API", version="1.0.0")

# ----- METRICS -----
REQUESTS = Counter("prediction_requests_total", "Total prediction requests")
ANOMALIES = Counter("anomalies_total", "Total detected anomalies")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency seconds")



# ----- INPUT SCHEMAS -----
class SensorRecord(BaseModel):
    """
    Schema representing a single sensor record.

    Attributes:
        temperature (float): Measured temperature in degrees Celsius.
        humidity (float): Measured humidity percentage.
        sound_volume (float): Measured sound volume in dB.
        timestamp (Optional[float]): Unix timestamp of measurement.
    """
    temperature: float = Field(..., example=25.0)
    humidity: float = Field(..., example=40.0)
    sound_volume: float = Field(..., example=55.0)
    timestamp: Optional[float] = Field(None, example=1630000000.0)

class BatchInput(BaseModel):
    """
    Schema representing a batch of sensor records.

    Attributes:
        records (List[SensorRecord]): List of sensor records to be processed.
    """
    records: List[SensorRecord] = Field(..., min_length=1)



# --- Preprocessing functions --- 
def any_values_beyond_range(table_row: pd.core.series.Series) -> bool:
    """
    Checks whether a row of sensor data contains any values beyond pre-defined limits.

    Args:
        table_row (pd.Series): Row of sensor data with 'temperature', 'humidity', 'sound_volume'.

    Returns:
        bool: 1 if any value exceeds defined min/max limits, otherwise 0.
    """
    current_temperature = int(table_row["temperature"])
    current_sound_volume = int(table_row["sound_volume"])
    current_humidity = int(table_row["humidity"])

    temperature_delta_pos = max_temperature - current_temperature
    temperature_delta_neg = current_temperature - min_temperature
    sound_delta_pos = max_sound_volume - current_sound_volume
    sound_delta_neg = current_sound_volume - min_sound_volume
    humidity_delta_pos = max_humidity - current_humidity
    humidity_delta_neg = current_humidity - min_humidity

    if any(delta < 0 for delta in [temperature_delta_pos, temperature_delta_neg, 
                                        sound_delta_pos, sound_delta_neg, 
                                        humidity_delta_pos, humidity_delta_neg]):
        return 1
    else:
        return 0

def clean_new_data(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Performs basic cleaning of a dataset: ensures correct columns and validates for NaNs.

    Args:
        dataset (pd.DataFrame): Raw sensor data.
        features (list[str]): List of expected feature column names.

    Returns:
        pd.DataFrame: Cleaned dataset containing only the specified features.

    Raises:
        ValueError: If any missing values are found in the input dataset.
    """
    # ensure correct columns
    cleaned_data = dataset[features].copy()

    # NaN handling: reject or fill — here we raise if NaN because streaming should provide valid measurements
    if cleaned_data.isnull().any().any():
        raise ValueError("Missing values in input")
    
    return cleaned_data
    
def preprocess_new_data(new_samples_dataset: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Preprocesses new sensor data: scales features and adds an 'is_beyond_range' flag.

    Args:
        new_samples_dataset (pd.DataFrame): Raw dataset to preprocess.
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler object.

    Returns:
        pd.DataFrame: Preprocessed dataset ready for inference, including:
                      - scaled features
                      - 'is_beyond_range' flag
    """
    feature_names = ["temperature", "sound_volume", "humidity"]
    cleaned_data = clean_new_data(dataset = new_samples_dataset, features = feature_names)
    scaled_data = scaler.fit_transform(cleaned_data[feature_names])
    preprocessed_data = pd.DataFrame(scaled_data, columns=feature_names)
    preprocessed_data["is_beyond_range"] = new_samples_dataset.apply(any_values_beyond_range, axis = 1)
    return preprocessed_data



# ----- HELPERS -----
def run_inference(model, preprocessed_data: np.ndarray):
    """
    Runs inference on preprocessed data using the given classification model.

    Args:
        model: Trained classification model with predict and predict_proba methods.
        preprocessed_data (np.ndarray or pd.DataFrame): Preprocessed feature matrix.

    Returns:
        tuple:
            flags (list[int]): Binary predictions for each sample (0=normal, 1=anomaly).
            scores (list[float]): Probability score for the positive class (anomaly) for each sample.
            probs (list[list[float]]): Full class probabilities for each sample.
    """
    # For classifier: probability for class 1 (anomaly)
    probs = model.predict_proba(preprocessed_data)  # shape (n, 2)
    preds = model.predict(preprocessed_data)
    # Build anomaly score as probability of positive class, and flag
    scores = probs[:, 1].tolist()
    flags = preds.tolist()
    return flags, scores, probs.tolist()

# ----- ENDPOINTS -----
@app.post("/predict")
def predict_single(record: SensorRecord):
    """
    Predicts whether a single sensor record is anomalous.

    Args:
        record (SensorRecord): Single sensor measurement.

    Returns:
        dict: {
            "input": input data,
            "is_anomaly": bool flag,
            "anomaly_score": float probability of anomaly,
            "probabilities": dict of class probabilities,
            "model_version": str
        }
    """
    REQUESTS.inc()
    start = time.time()
    try:
        new_data = pd.DataFrame([record.dict()])
        preprocessed_data = preprocess_new_data(new_data, scaler)
        flags, scores, probs = run_inference(model = logistic_regression, 
                                             preprocessed_data = preprocessed_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="internal server error")

    latency = time.time() - start
    LATENCY.observe(latency)

    if flags[0] == 1:
        ANOMALIES.inc()

    return {
        "input": record.dict(),
        "is_anomaly": bool(flags[0]),
        "anomaly_score": float(scores[0]),
        "probabilities": {"class_0": probs[0][0], "class_1": probs[0][1]},
        "model_version": "1.0.0"
    }

@app.post("/predict_batch")
def predict_batch(payload: BatchInput):
    """
    Predicts anomalies for a batch of sensor records.

    Args:
        payload (BatchInput): Batch of sensor records.

    Returns:
        dict: {
            "model_version": str,
            "results": list of prediction dictionaries for each record
        }
    """
    REQUESTS.inc()
    start = time.time()
    try:
        rows = [r.model_dump() for r in payload.records]
        new_data = pd.DataFrame(rows)
        preprocessed_data = preprocess_new_data(new_data, scaler)
        flags, scores, probs = run_inference(model = logistic_regression, 
                                             preprocessed_data = preprocessed_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="internal server error")

    latency = time.time() - start
    LATENCY.observe(latency)

    # update anomaly counter
    for f in flags:
        if f == 1:
            ANOMALIES.inc()

    results = []
    for inp, f, s, p in zip(rows, flags, scores, probs):
        results.append({
            "input": inp,
            "is_anomaly": bool(f),
            "anomaly_score": float(s),
            "probabilities": {"class_0": p[0], "class_1": p[1]}
        })
    return {"model_version": "1.0.0", "results": results}

@app.get("/health")
def health():
    """
    Returns the health status of the API and model version.

    Returns:
        dict: {"status": "ok", "model_version": str}
    """
    return {"status": "ok", "model_version": "1.0.0"}

@app.get("/metrics")
def metrics():
    """
    Returns Prometheus metrics for monitoring.

    Returns:
        fastapi.Response: Metrics in Prometheus exposition format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



# LINES TO ACTIVATE LOCALY:
# uvicorn IntegrationFile:app --reload --host 0.0.0.0 --port 8000

# To run the stream (open cmd, switch the directory and run the following line):
# python StreamImitation.py