import random
import csv



class DataGenerator:
    """
    A class for generating synthetic sensor data with optional anomalies.

    Attributes:
        __min_temperature (float): Minimum normal temperature value.
        __max_temperature (float): Maximum normal temperature value.
        __min_humidity (float): Minimum normal humidity value.
        __max_humidity (float): Maximum normal humidity value.
        __min_sound_volume (float): Minimum normal sound volume value.
        __max_sound_volume (float): Maximum normal sound volume value.
        __duration_in_seconds (int): Total duration for data generation in seconds.
        __step_in_seconds (int): Time step between generated data points in seconds.
        __anomaly_rate (float): Probability that a generated row is anomalous (0 to 1).
        __generated_data_path (str): File path to save the generated CSV dataset.
    """
    
    def __init__(self, min_temperature: float, max_temperature: float,
                 min_humidity: float, max_humidity: float,
                 min_sound_volume: float, max_sound_volume: float,
                 duration_in_seconds: int, step_in_seconds: int,
                 anomaly_rate: float, generated_data_path: str):
        """
        Initializes the DataGenerator with specified ranges and parameters.

        Args:
            min_temperature (float): Minimum normal temperature.
            max_temperature (float): Maximum normal temperature.
            min_humidity (float): Minimum normal humidity.
            max_humidity (float): Maximum normal humidity.
            min_sound_volume (float): Minimum normal sound volume.
            max_sound_volume (float): Maximum normal sound volume.
            duration_in_seconds (int): Duration for dataset generation in seconds.
            step_in_seconds (int): Step between data points in seconds.
            anomaly_rate (float): Probability of generating an anomalous row (0 to 1).
            generated_data_path (str): Path to save the generated CSV file.
        """
        self.__min_temperature = min_temperature
        self.__max_temperature = max_temperature
        self.__min_humidity = min_humidity
        self.__max_humidity = max_humidity
        self.__min_sound_volume = min_sound_volume
        self.__max_sound_volume = max_sound_volume

        self.__duration_in_seconds = duration_in_seconds
        self.__step_in_seconds = step_in_seconds
        self.__anomaly_rate = anomaly_rate
        self.__generated_data_path = generated_data_path



    def generate_outlier(self, min_value: float, max_value) -> float:
        """
        Generates an outlier value outside the given min and max range.

        Args:
            min_value (float): Lower bound for normal values.
            max_value (float): Upper bound for normal values.

        Returns:
            float: A value either below min_value or above max_value.
        """
        if random.random() < 0.5:
            return min_value - random.uniform(5, 15)
        else:
            return max_value + random.uniform(5, 15)

    def generate_dataset_row(self) -> dict[str, object]:
        """
        Generates a single row of sensor data, optionally as an anomaly.

        Returns:
            dict[str, object]: Dictionary containing generated sensor values and 
                               an 'is_anomaly' flag (0 or 1).
        """
        is_anomaly = random.random() < self.__anomaly_rate

        if not is_anomaly:
            current_temperature = random.uniform(self.__min_temperature, self.__max_temperature)
            current_humidity = random.uniform(self.__min_humidity, self.__max_humidity)
            current_sound_volume = random.uniform(self.__min_sound_volume, self.__max_sound_volume)
        else:
            current_temperature = self.generate_outlier(self.__min_temperature, self.__max_temperature)                
            current_humidity = self.generate_outlier(self.__min_humidity, self.__max_humidity)
            current_sound_volume = self.generate_outlier(self.__min_sound_volume, self.__max_sound_volume)

        generated_row = {
            "temperature": current_temperature, 
            "sound_volume": current_sound_volume, 
            "humidity": current_humidity, 
            "is_anomaly": int(is_anomaly)
        }
        return generated_row

    def generate_dataset(self) -> None:
        """
        Generates a complete dataset and writes it to a CSV file at the specified path.

        Each row corresponds to a time step and includes temperature, humidity, 
        sound volume, and anomaly flag.

        Returns:
            None
        """
        titles = ["temperature", "sound_volume", "humidity", "is_anomaly"]
        with open(self.__generated_data_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=titles)
            writer.writeheader()

            for i in range(0, self.__duration_in_seconds, self.__step_in_seconds):
                generated_row = self.generate_dataset_row()
                # generated_row["time"] = i+1
                writer.writerow(generated_row)

