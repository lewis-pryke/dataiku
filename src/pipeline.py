from model_trainer import ModelTrainer

from typing import List

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score


class Pipeline:
    """Pipeline for model"""

    def __init__(self, label_col: str, seed: int, models: dict) -> None:
        self._label_col = label_col
        self._seed = seed
        self._models = models
 
    def run(self):
        """
        1. Get the learning and test data
        2. Preprocess the data
        3. Train and evaluate the models
        4. Compare model performance and select best
        5. Generate predictions for test data

        :return: f1 score on unseen test data
        """
        # get data
        # manually create col_names.csv as learn/test files came with no column names
        col_names = pd.read_csv("~/DS/dataiku/data/col_names.csv", header=None)
        learning = pd.read_csv("~/DS/dataiku/data/census_income_learn.csv", names=col_names[0].values.tolist())
        test = pd.read_csv("~/DS/dataiku/data/census_income_test.csv", names=col_names[0].values.tolist())

        # preprocess the data
        preprocessed_train = self.preprocess(df=learning)
        preprocessed_test = self.preprocess(df=test)

        # manual selection to encode based on EDA
        cat_cols = self.get_cat_features()
        # encoded column names
        ohe_train, ohe_test, cat_cols_encoded = self.one_hot_encoder(train_df=preprocessed_train, 
                                                                     test_df=preprocessed_test, 
                                                                     cols=cat_cols)
        
        features = cat_cols_encoded.tolist() + self.get_num_features()
        # save preprocessed train and test sets with features & target only
        ohe_train[features + ['target']].to_csv('~/DS/dataiku/data/train.csv', index=False)
        
        X_train, X_val, y_train, y_val = self.create_train_val_sets(df=ohe_train, 
                                                                    train_split=0.8, 
                                                                    features=features)
    
        # initialise ModelTrainer with models
        trainer = ModelTrainer(self._models)

        # train models
        trainer.train(X_train, y_train)
        
        # produce predictions for validation set
        probabilities = trainer.predict_probabilities(X_val)

        # evaluate models
        # threshold selection here is just to get an initial reading of performance
        # optimal threshold is selected later
        evaluation_results = trainer.evaluate(y_val, probabilities, threshold=0.1)

        # print evaluation results
        for model_name, metrics in evaluation_results.items():
            print(f"Model: {model_name}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
        
        # get the best model and its optimal threshold using validation data
        best_model, best_threshold = trainer.get_best_model(y_val, probabilities)

        # print the best model and threshold
        print(f"Best Model: {best_model.__class__.__name__}")
        print(f"Optimal Threshold: {best_threshold}")

        # PREDICTIONS
        # make predictions using the best model and threshold on train set
        train_probs = best_model.predict_proba(ohe_train[features])[:, 1]
        train_preds = (train_probs > best_threshold).astype(int)
        ohe_train['prob'] = train_probs
        ohe_train['prediction'] = train_preds
        ohe_train.to_csv('~/DS/dataiku/data/train_predictions.csv', index=False)

        # make predictions using the best model and threshold on test set
        # TODO: should detect drift between training data and test data before producing predictions
        test_probs = best_model.predict_proba(ohe_test[features])[:, 1]
        test_preds = (test_probs > best_threshold).astype(int)
        ohe_test['prob'] = test_probs
        ohe_test['prediction'] = test_preds
        ohe_test.to_csv('~/DS/dataiku/data/test_predictions.csv', index=False)

        # evaluate test set results
        test_f1 = f1_score(ohe_test[self._label_col], test_preds)
        
        print(f"F1 score using best model on Unseen Test data: {test_f1}")

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Clean column names
        2. Drop duplicates for data
        3. Binarize label column & skewed continuous columns
        """
        df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
        df = df.drop_duplicates()
        dd = df.copy()
        dd['target'] = dd['target'].replace(' - 50000.', 0)
        dd['target'] = dd['target'].replace(' 50000+.', 1)

        # binarize skewed numerical cols
        for col in ['capital_gains', 'capital_losses', 'divdends_from_stocks']:
            dd[f'{col}_grp'] = np.where(dd[col]>0, 1, 0)

        return dd

    @staticmethod
    def one_hot_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> (pd.DataFrame, pd.DataFrame, np.array):
        """
        Categorical features to be encoded so models can use them
        Encoder fitted on train set and applied to train and test
        """
        # keep string columns
        cat_cols = train_df[cols].select_dtypes(exclude=[np.number]).columns.to_list()

        # fit encoder on train set - only if more than 1% of values exist
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', min_frequency=0.01)
        encoder.fit(train_df[cat_cols])
        
        # transform train and validation sets
        train = pd.DataFrame(encoder.transform(train_df[cat_cols]), columns=encoder.get_feature_names_out())
        tst = pd.DataFrame(encoder.transform(test_df[cat_cols]), columns=encoder.get_feature_names_out())

        # align indexes
        train.index, tst.index = train_df.index, test_df.index

        # df of numeric cols
        num_train = train_df.drop(cat_cols, axis=1)
        num_tst = test_df.drop(cat_cols, axis=1)

        # Add one-hot encoded columns to numerical features
        ohe_train = pd.concat([num_train, train], axis=1)
        ohe_tst = pd.concat([num_tst, tst], axis=1)

        return ohe_train, ohe_tst, encoder.get_feature_names_out()

    @staticmethod
    def get_num_features() -> List[str]:
        """Manual selection of numerical features based on EDA"""
        return ['age', 'weeks_worked_in_year', 'num_persons_worked_for_employer']

    @staticmethod
    def get_cat_features() -> List[str]:
        """Manual selection of categorical features based on EDA"""
        return ['class_of_worker',
                'education',
                'marital_status',
                'sex',
                'member_of_a_labor_union',
                'major_occupation_code',
                'major_industry_code',
                'race',
                'capital_gains_grp',
                'capital_losses_grp', 
                'divdends_from_stocks_grp']

    def create_train_val_sets(self, df: pd.DataFrame, train_split: float, features: List[str]):
        """Create train/test sets using shuffle split to keep the same label distribution"""
        X = df[features].values
        y = df[self._label_col].values

        sss = StratifiedShuffleSplit(n_splits=2, train_size=train_split, random_state=self._seed)
        train_index, val_index = next(sss.split(X, y))

        return X[train_index], X[val_index], y[train_index], y[val_index]
    