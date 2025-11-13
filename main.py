import json
import os
import pandas as pd

from DataGenerator import DataGenerator
from DataPreprocessor import DataPreprocessor
from AnomalyClassifier import AnomalyClassifier
from sklearn.model_selection import train_test_split

os.makedirs("./datasets", exist_ok = True)
os.makedirs("./models", exist_ok = True)

with open('config.json', 'r') as f:
    config = json.load(f)

min_temperature = config["DatasetGeneratorParameters"]["min_temperature"]
max_temperature = config["DatasetGeneratorParameters"]["max_temperature"]
min_humidity = config["DatasetGeneratorParameters"]["min_humidity"]
max_humidity = config["DatasetGeneratorParameters"]["max_humidity"]
min_sound_volume = config["DatasetGeneratorParameters"]["min_sound_volume"]
max_sound_volume = config["DatasetGeneratorParameters"]["max_sound_volume"]
duration_in_seconds = config["DatasetGeneratorParameters"]["duration_in_seconds"]
step_in_seconds = config["DatasetGeneratorParameters"]["step_in_seconds"]
anomaly_rate = config["DatasetGeneratorParameters"]["anomaly_rate"]

generated_data_path = config["DatasetPaths"]["generated_data_path"]
preprocessed_data_path = config["DatasetPaths"]["preprocessed_data_path"]
path_to_scaler = config["DataPerprocessorParameters"]["path_to_scaler"]

test_size = config["AnomalyClassifierParameters"]["test_size"]
random_state_split = config["AnomalyClassifierParameters"]["random_state_split"]
logistic_regression_path = config["AnomalyClassifierParameters"]["logistic_regression_path"]
logistic_regression_random_state = config["AnomalyClassifierParameters"]["logistic_regression_random_state"]



if __name__ == '__main__':
    # Generation of synthetic data
    data_generator = DataGenerator(min_temperature = min_temperature, max_temperature = max_temperature,
                                   min_humidity = min_humidity, max_humidity = max_humidity,
                                   min_sound_volume = min_sound_volume, max_sound_volume = max_sound_volume,
                                   duration_in_seconds = duration_in_seconds, step_in_seconds = step_in_seconds,
                                   anomaly_rate = anomaly_rate, generated_data_path = generated_data_path)
    data_generator.generate_dataset()

    # Data analysis and preprocessing
    data_preprocessor = DataPreprocessor(min_temperature = min_temperature, max_temperature = max_temperature,
                                         min_humidity = min_humidity, max_humidity = max_humidity,
                                         min_sound_volume = min_sound_volume, max_sound_volume = max_sound_volume)
    data_preprocessor.load_raw_data(raw_data_path = generated_data_path)
    data_preprocessor.analyze_raw_data()
    data_preprocessor.preprocess_raw_data()
    data_preprocessor.save_preprocessed_data(preprocessed_data_path = preprocessed_data_path)
    data_preprocessor.save_scaler(path_to_scaler = path_to_scaler)

    # Training of an anomaly classifiers and saving the results
    anomaly_classifier = AnomalyClassifier()
    data = pd.read_csv(preprocessed_data_path)
    x = data.drop("is_anomaly", axis = 1)
    y = data["is_anomaly"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size = test_size,
                                                        random_state = random_state_split,
                                                        stratify = y)
    
    os.makedirs(logistic_regression_path, exist_ok=True)
    anomaly_classifier.train_logistic_regression(x_train = x_train, y_train = y_train,
                                                 x_test = x_test, y_test = y_test,
                                                 path_to_results = logistic_regression_path,
                                                 random_state = logistic_regression_random_state)
    anomaly_classifier.save_models()



