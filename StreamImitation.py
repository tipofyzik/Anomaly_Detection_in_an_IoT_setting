"""
Simulates real-time sensor data streaming to the Anomaly Detection API.

This script repeatedly generates random sensor measurements and sends them
as POST requests to the `/predict` endpoint of a locally running FastAPI server.

The script mimics a streaming scenario with a configurable rate.

Requirements:
    - requests
    - random
    - time

Usage:
    python StreamImitation.py
"""



import time, requests, random

predict_batch = False
URL = "http://localhost:8000/predict"
if predict_batch:
    URL = "http://localhost:8000/predict_batch"

def generate_random_payload() -> dict[str, float]:
    """
    Generates a random sensor measurement payload.

    Returns:
        dict[str, float]: Dictionary containing:
            - temperature (float): Random temperature between 60 and 80
            - humidity (float): Random humidity between 40 and 60
            - sound_volume (float): Random sound volume between 50 and 70
    """
    return {
        "temperature": random.uniform(60, 80),
        "humidity": random.uniform(40, 60),
        "sound_volume": random.uniform(50, 70)
    }

def main(stream_interval: float = 0.5) -> None:
    """
    Continuously sends random sensor measurements to the prediction API.

    Args:
        stream_interval (float, optional): Time interval between requests in seconds.
                                           Defaults to 0.5 (2 requests/sec).

    Returns:
        None
    """
    while True:
        payload = generate_random_payload()
        if predict_batch:
            payload = {"records": [payload]}
        try:
            response = requests.post(URL, json=payload, timeout=5)
            print(response.json())
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(stream_interval)



if __name__ == "__main__":
    main()

