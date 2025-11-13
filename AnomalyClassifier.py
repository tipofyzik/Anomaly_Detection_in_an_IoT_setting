import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from collections import defaultdict
import seaborn as sns
import pandas as pd
import numpy as np
import joblib


class AnomalyClassifier:
    """
    A class for training, evaluating, and saving anomaly detection models.
    
    Attributes:
        __models (defaultdict): Dictionary storing trained models for each result path.
    """

    def __init__(self):
        """
        Initializes the AnomalyClassifier with an empty dictionary for storing models.
        """
        self.__models = defaultdict(dict)

    def train_logistic_regression(self, x_train: pd.Series, x_test: pd.Series, 
                      y_train: pd.Series, y_test: pd.Series,
                      path_to_results: str, random_state: int, 
                      i: int = 0) -> None:
        """
        Trains ligistic regression model and saves the confusion matrix and the report with various metrics, 
        such as accuracy, precision, f1-score, etc.

        Args: 
            x_train (pd.Series): The train part of features.
            x_test (pd.Series): The test part of features.
            y_train (pd.Series): The train part of the target variable.
            y_test (pd.Series): The test part of the target variable.
            path_to_results (str): The path where the result model will be saved.
            random_state (int): The random state for model initialization.
            i (int): The number the reflects the function call number (to gain statistics).
        """
        logistic_regression = LogisticRegression(C=1.0, 
                                                 max_iter=1000, 
                                                 random_state = random_state, 
                                                 solver='liblinear')
        logistic_regression.fit(x_train, y_train)
        y_pred = logistic_regression.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.__save_classification_report(pd.DataFrame(report).transpose().round(3), 
                                        path_to_results, 
                                        filename = f"logistic_regression_report_{i}")
        self.__save_confusion_matrix(confusion_matrix(y_test, y_pred),
                                    path_to_results,
                                    filename = f"logistic_regression_confusion_matrix_{i}")
        self.__models[path_to_results]["logistic_regression"] = logistic_regression



    def __save_classification_report(self, report: pd.DataFrame, path_to_results: str, filename: str):
        """
        Saves the classification report as an image.

        Args: 
            report (pd.DataFrame): The report that contains the following metrics: presicion, recall, f1-score,
            accuracy, macro and weighted average scores.
            path_to_results (str): The path where the classification report will be saved.
            filename (str): The name of the final image.
        """
        _, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        table = ax.table(cellText=report.values, colLabels=report.columns, rowLabels=report.index, loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.text(0.5, 0.7, filename, fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

        plt.savefig(f"{path_to_results}/{filename}.jpg", bbox_inches='tight', dpi=300)
        plt.close()

    def __save_confusion_matrix(self, confusion_matrix: np.ndarray, path_to_results: str, filename: str):
        """
        Saves the confusion matrix as an image.

        Args: 
            confusion_matrix (np.ndarray): The confusion_matrix.
            path_to_results (str): The path where the confusion_matrix will be saved.
            filename (str): The name of the final image.
        """
        _, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=False, fmt='d', cmap='Blues',
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    cbar=False, ax=ax)

        # Adding annotation:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j + 0.5, i + 0.5, str(confusion_matrix[i, j]),
                        ha='center', va='center', color='black', fontsize=12)

        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(filename)
        ax.tick_params(length=0)

        plt.savefig(f"{path_to_results}/{filename}.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    def save_models(self):
        """
        Saves all trained models to their respective paths as .pkl files using joblib.
        """
        for path_to_results, inner_dict in self.__models.items():
            for model_name, model in inner_dict.items():
                joblib.dump(model, f"{path_to_results}/{model_name}.pkl")