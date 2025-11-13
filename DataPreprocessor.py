from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import pandas as pd


class DataPreprocessor:
    """
    A class for loading, analyzing, preprocessing, and saving sensor data.

    Attributes:
        __raw_data (pd.DataFrame): Raw dataset loaded from CSV.
        __preprocessed_data (pd.DataFrame): Preprocessed dataset after scaling and feature engineering.
        __scaler (StandardScaler): Scaler used to standardize numerical features.
        __min_temperature (float): Minimum allowed temperature for range checking.
        __max_temperature (float): Maximum allowed temperature for range checking.
        __min_humidity (float): Minimum allowed humidity for range checking.
        __max_humidity (float): Maximum allowed humidity for range checking.
        __min_sound_volume (float): Minimum allowed sound volume for range checking.
        __max_sound_volume (float): Maximum allowed sound volume for range checking.
    """
    
    def __init__(self, min_temperature: float, max_temperature: float,
                 min_humidity: float, max_humidity: float,
                 min_sound_volume: float, max_sound_volume: float):
        """
        Initializes the DataPreprocessor with range limits for sensor values.

        Args:
            min_temperature (float): Minimum temperature value.
            max_temperature (float): Maximum temperature value.
            min_humidity (float): Minimum humidity value.
            max_humidity (float): Maximum humidity value.
            min_sound_volume (float): Minimum sound volume value.
            max_sound_volume (float): Maximum sound volume value.
        """
        self.__raw_data = pd.DataFrame()
        self.__preprocessed_data = pd.DataFrame()
        self.__scaler = StandardScaler()

        self.__min_temperature = min_temperature
        self.__max_temperature = max_temperature
        self.__min_humidity = min_humidity
        self.__max_humidity = max_humidity
        self.__min_sound_volume = min_sound_volume
        self.__max_sound_volume = max_sound_volume


    
    def load_raw_data(self, raw_data_path: str) -> None:
        """
        Loads a raw dataset from a CSV file into the internal DataFrame.

        Args:
            raw_data_path (str): Path to the raw CSV file.

        Returns:
            None
        """
        self.__raw_data = pd.read_csv(raw_data_path)

    def analyze_raw_data(self) -> None:
        """
        Prints general information about the raw dataset, including
        column names, data types, null counts, and dataset shape.

        Returns:
            None
        """
        info = pd.DataFrame({
            "column": self.__raw_data.columns,
            "dtype": self.__raw_data.dtypes.values,
            "null_count": self.__raw_data.isnull().sum().values,
        })
        print(f"Dataset general data:\n{info.to_string(justify='left', index=False)}")
        print(f"Dataset shape: {self.__raw_data.shape}")



    def any_values_beyond_range(self, table_row: pd.core.series.Series) -> bool:
        """
        Checks if any sensor values in a row exceed the defined min/max ranges.

        Args:
            table_row (pd.Series): A single row of the dataset.

        Returns:
            bool: True (1) if any value is beyond range, False (0) otherwise.
        """
        current_temperature = table_row["temperature"]
        current_sound_volume = table_row["sound_volume"]
        current_humidity = table_row["humidity"]

        temperature_delta_pos = self.__max_temperature - current_temperature
        temperature_delta_neg = current_temperature - self.__min_temperature
        sound_delta_pos = self.__max_sound_volume - current_sound_volume
        sound_delta_neg = current_sound_volume - self.__min_sound_volume
        humidity_delta_pos = self.__max_humidity - current_humidity
        humidity_delta_neg = current_humidity - self.__min_humidity

        if any(delta < 0 for delta in [temperature_delta_pos, temperature_delta_neg, 
                                           sound_delta_pos, sound_delta_neg, 
                                           humidity_delta_pos, humidity_delta_neg]):
            return 1
        else:
            return 0

    def preprocess_raw_data(self) -> None:
        """
        Scales the feature columns (temperature, sound volume, humidity),
        adds a flag for out-of-range values, and preserves the original anomaly flag.

        Returns:
            None
        """
        self.__feature_names = ["temperature", "sound_volume", "humidity"]
        scaled_data = self.__scaler.fit_transform(self.__raw_data[self.__feature_names])
        self.__preprocessed_data = pd.DataFrame(scaled_data, columns=self.__feature_names)
        self.__preprocessed_data["is_beyond_range"] = self.__raw_data.apply(self.any_values_beyond_range, axis = 1)
        self.__preprocessed_data["is_anomaly"] = self.__raw_data["is_anomaly"]



    def save_preprocessed_data(self, preprocessed_data_path: str) -> None:
        """
        Saves the preprocessed dataset to a CSV file.

        Args:
            preprocessed_data_path (str): Path to save the CSV file.

        Returns:
            None
        """
        self.__preprocessed_data.to_csv(preprocessed_data_path, index = False)

    def save_scaler(self, path_to_scaler):
        """
        Saves the fitted scaler to disk using joblib.

        Args:
            path_to_scaler (str): File path to save the scaler.

        Returns:
            None
        """
        joblib.dump(self.__scaler, path_to_scaler)
