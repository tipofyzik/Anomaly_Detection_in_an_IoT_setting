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

predict_batch = True
URL = "http://localhost:8000/predict"
if predict_batch:
    URL = "http://localhost:8000/predict_batch"

Health_URL = "http://localhost:8000/health"
Metrics_URL = "http://localhost:8000/metrics"
batch_size = 3

def generate_random_payload(batch_size = batch_size) -> list[dict[str, float]]:
    """
    Generates a random sensor measurement payload.

    Args:
        batch_size (int, optional): Number of records to generate for batch prediction.

    Returns:
        list[dict[str, float]]: List of dictionaries, each containing:
            - temperature (float): Random temperature between 60 and 80
            - humidity (float): Random humidity between 40 and 60
            - sound_volume (float): Random sound volume between 50 and 70
    """
    return [{
        "temperature": random.Random().uniform(63, 78),
        "humidity": random.Random().uniform(43, 58),
        "sound_volume": random.Random().uniform(53, 68)} 
        for _ in range(batch_size)]

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
            payload = {"records": payload}
        try:
            response = requests.post(URL, json=payload, timeout=5)
            data = response.json()

            # Print the response in a readable format
            if predict_batch:
                print(f"\n--- Batch of {len(data['results'])} records ---")
                for i, rec in enumerate(data["results"], start=1):
                    print(f"Record {i}: Temperature={rec['input']['temperature']:.2f}, "
                          f"Humidity={rec['input']['humidity']:.2f}, "
                          f"Sound={rec['input']['sound_volume']:.2f}, "
                          f"Anomaly={rec['is_anomaly']}, Anomaly score={rec['anomaly_score']:.3f}")
            else:
                rec = data
                print(f"Temperature={rec['input']['temperature']:.2f}, "
                      f"Humidity={rec['input']['humidity']:.2f}, "
                      f"Sound={rec['input']['sound_volume']:.2f}, "
                      f"Anomaly={rec['is_anomaly']}, Anomaly score={rec['anomaly_score']:.3f}")

        except requests.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(stream_interval)



if __name__ == "__main__":
    main()

