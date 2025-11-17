import joblib
import json
import pandas as pd



with open('config.json', 'r') as f:
    config = json.load(f)

logistic_regression_path = config["AnomalyClassifierParameters"]["logistic_regression_path"]
path_to_scaler = config["DataPerprocessorParameters"]["path_to_scaler"]

min_temperature = config["DatasetGeneratorParameters"]["min_temperature"]
max_temperature = config["DatasetGeneratorParameters"]["max_temperature"]
min_humidity = config["DatasetGeneratorParameters"]["min_humidity"]
max_humidity = config["DatasetGeneratorParameters"]["max_humidity"]
min_sound_volume = config["DatasetGeneratorParameters"]["min_sound_volume"]
max_sound_volume = config["DatasetGeneratorParameters"]["max_sound_volume"]



logistic_regression = joblib.load(f"{logistic_regression_path}/logistic_regression.pkl")
scaler = joblib.load(path_to_scaler)



def any_values_beyond_range(table_row: pd.core.series.Series) -> bool:
    """
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

def preprocess_new_data(new_samples_dataset: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    """
    feature_names = ["temperature", "sound_volume", "humidity"]
    scaled_data = scaler.fit_transform(new_samples_dataset[feature_names])
    preprocessed_data = pd.DataFrame(scaled_data, columns=feature_names)
    preprocessed_data["is_beyond_range"] = new_samples_dataset.apply(any_values_beyond_range, axis = 1)
    return preprocessed_data



new_samples = pd.DataFrame({
    "temperature": [20, 70, 30],
    "humidity": [40, 50, 60],
    "sound_volume": [50, 55, 65]
})

preprocessed_new_data = preprocess_new_data(new_samples_dataset = new_samples,
                                            scaler = scaler)

# Предсказание
predictions = logistic_regression.predict(preprocessed_new_data)
probabilities = logistic_regression.predict_proba(preprocessed_new_data)

for prediction, probability in zip(predictions, probabilities):
    print("Prediction:", prediction)
    print("Probability:", probability)