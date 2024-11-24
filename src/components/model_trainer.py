import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import ( 
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor, 
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from src.utlis import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Evaluate multiple models and return a performance report.
        Args:
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            models: A dictionary of model names and their instances.
        Returns:
            A dictionary with model names as keys and a dictionary of train/test scores as values.
        """
        report = {}
        for model_name, model_instance in models.items():
            try:
                logging.info(f"Evaluating model: {model_name}")
                model_instance.fit(X_train, y_train)
                train_model_score = model_instance.score(X_train, y_train)
                test_model_score = model_instance.score(X_test, y_test)
                report[model_name] = {
                    "train_score": train_model_score,
                    "test_score": test_model_score,
                }
            except Exception as e:
                raise CustomException(f"Error evaluating model {model_name}: {str(e)}", sys)
        return report

    def inititate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
            }

            # Evaluate models and get report
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model Report: {model_report}")

            # Find the best model by test score
            best_model_name = max(model_report, key=lambda model: model_report[model]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]

            # Select the best model
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient performance.")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Predict and calculate r2_score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
