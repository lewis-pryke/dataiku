from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib 
import os


class ModelTrainer:
    def __init__(self, models):
        self.models = models

    def train(self, X_train, y_train):
        """ Fit the models from the ModelTrainer instance and add to self.trained_models  """
        self.trained_models = {}
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model

    def predict_probabilities(self, X_test):
        """ Generate predictions for models trained in self.trained_models """
        probabilities = {}
        for model_name, model in self.trained_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]  # probability of positive class
                probabilities[model_name] = proba
            else:
                raise ValueError(f"{model_name} does not support predict_proba method.")
        return probabilities

    @staticmethod
    def evaluate(y_true, probabilities, threshold):
        """ 
        For models used for predictions, evaluate based on threshold
        Args:
            y_true: true value of y 
            probabilities: probability of outcome
            threshold: used to turn probability into binary prediction
        Returns:
            Dictionary of results for each model
        """
        evaluation_results = {}
        for model_name, proba in probabilities.items():
            y_pred = (proba > threshold).astype(int)

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, proba)

            evaluation_results[model_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
            }
        return evaluation_results

    @staticmethod
    def find_optimal_threshold(y_true, probabilities):
        """
        For a range of thresholds, find optimal threshold to maximise f1-score
        Args:
            y_true: true value of y
            probabilities: predicted probability of y
        Returns:
            Optimal thresholds for each model
        """
        optimal_thresholds = {}
        for model_name, proba in probabilities.items():
            thresholds = np.linspace(0, 1, 100)  # Range of threshold values to try
            f1_scores = []
            for threshold in thresholds:
                y_pred = (proba > threshold).astype(int)
                f1 = f1_score(y_true, y_pred)
                f1_scores.append(f1)
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            optimal_thresholds[model_name] = optimal_threshold
        return optimal_thresholds
    
    def get_best_model(self, y_true, probabilities):
        """
        Find and save best model based on optimal threshold
        """
        optimal_thresholds = self.find_optimal_threshold(y_true, probabilities)
        best_model_name = None
        best_f1_score = 0
        for model_name, threshold in optimal_thresholds.items():
            y_pred = (probabilities[model_name] > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_name = model_name
        
        # save best model
        path = "saved_models/best_model.joblib"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.trained_models[best_model_name], path)
        print(f"Save best model {best_model_name} to {path}")

        return self.trained_models[best_model_name], optimal_thresholds[best_model_name]

    def tune_hyperparameters(self, model, train_df):
        """
        Placeholder to tune hyperparamters for models, rather than using default params
        Can help avoid overfitting and should improve performance
        """
        pass

    def baseline_model(self):
        """
        Placeholder to create simple model that more complex models would have to beat to pass
        """
        pass
