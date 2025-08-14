import torch
import torch.nn as nn
import seaborn as sns
from matplotlib.colors          import LinearSegmentedColormap
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import optuna
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute          import SimpleImputer
from copy import deepcopy
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Intentar importar cuML
import cupy as cp

from cuml.ensemble import RandomForestClassifier as cuRF
from xgboost import XGBClassifier as cuXGB
import torch

# Imports MLP
from pytorch_pipeline import MLP, Pytorch_Pipeline


def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(x) for x in obj]
    elif isinstance(obj, np.generic):  # np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj

def get_metrics(y_true, y_pred, verbose = True):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'cm': confusion_matrix(y_true, y_pred)
    }
    if verbose:
      print(metrics)
    return metrics


class optuna_objective_cv_sklearn:
    def __init__(self, X, y, model_name, n_classes=2, SMOTE_on = None, sample_weights = None):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.n_classes = n_classes
        self.results = {}
        self.SMOTE_on = SMOTE_on
        self.sample_weights_on = sample_weights
        self.sample_weights = None

    def objective(self, trial):
        # Hiperparámetros comunes
        model_name = self.model_name
        params = self.get_params(trial, model_name)

        # Validación cruzada estratificada por grupo
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        F1 = []
        all_metrics = []

        for numfold, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]


            if self.SMOTE_on is not None:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            if self.sample_weights_on is not None:
              self.sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

            model, y_pred = self.train_model_GPU(model_name, X_train, y_train, X_test, params)


            f1 = f1_score(y_test, y_pred, average='weighted')
            F1.append(f1)

            metrics = get_metrics(y_test, y_pred)
            all_metrics.append(metrics)

        mean_f1 = np.mean(F1)

        if trial.number == 0 or mean_f1 > trial.study.best_value:
            self.results = {
                "model_name": model_name,
                'metrics': self.avg_metrics(all_metrics),
                "params": params,
                "mean_f1": mean_f1,
                "model": deepcopy(model)
            }

        return mean_f1

    def train_model_GPU(self, model_name, X_train, y_train, X_test, params):
        gpu_detected = torch.cuda.is_available()

        if model_name == "RandomForestClassifier":
            if gpu_detected:
                X_train, y_train, X_test = map(cp.asarray, (X_train, y_train, X_test))
                model = cuRF(**params)
                model.fit(X_train, y_train)
                y_pred = cp.asnumpy(model.predict(X_test))
            else:
                print("⚠️ Entrenando XGBoost en CPU...")
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train, sample_weight=self.sample_weights)
                y_pred = model.predict(X_test)

        elif model_name == "XGBClassifier":
            if gpu_detected:
                model = cuXGB(**params)
                model.fit(X_train, y_train, sample_weight=self.sample_weights)
                y_pred = model.predict(X_test)
            else:
                params.update({"tree_method": "hist"})
                print("⚠️ Entrenando XGBoost en CPU...")
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, sample_weight=self.sample_weights)
                y_pred = model.predict(X_test)

        elif model_name == "MLP":
            pipeline_mlp =  Pytorch_Pipeline(model_class=MLP, sample_weights_loss = self.sample_weights_loss)
            #Set params
            pipeline_mlp.set_params(**params)
            #Set criterion
            pipeline_mlp.set_criterion(y_train)
            

        else:
            model = self.model_class(**params)
            model.fit(X_train, y_train, sample_weight=self.sample_weights)
            y_pred = model.predict(X_test)

        return model, y_pred

    def avg_metrics(self, all_metrics):
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            values = [metrics[metric] for metrics in all_metrics]

            if metric == 'cm':
                avg_metrics[metric] = np.mean(values, axis=0).astype(int)  # o float si prefieres
            else:
                avg_metrics[metric] = np.mean(values)
        return avg_metrics

    def get_results(self):
        return self.results

    def get_params(self, trial, model_name):

        if model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                "max_iter": 1000
            }
        elif model_name == "DecisionTreeClassifier":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
        elif model_name == "RandomForestClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 20)
            }
        elif model_name == "GradientBoostingClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 2, 10)
            }
        elif model_name == "XGBClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 5, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eval_metric": "logloss"
            }
        elif model_name == "SVC":
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                "probability": True
            }
        elif model_name == "SGDClassifier":
            params = {
                "loss": trial.suggest_categorical("loss", ["hinge", "log_loss"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "max_iter": 1000,
                "tol": 1e-3
            }
        return params
    


n_classes = 3
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_classes=n_classes,
    #weights=[0.95],  # 90% clase 0, 10% clase 1
    random_state=42,
    shuffle=True
)


model_list = [
    #'LogisticRegression',
    #'DecisionTreeClassifier',
    #'RandomForestClassifier',
    #'GradientBoostingClassifier',
    'XGBClassifier',
    #'SVC',
    #'SGDClassifier'
]

all_results = {}

for name in model_list:
    print(f"🔍 Evaluando modelo: {name}")

    # Crear instancia
    objective = optuna_objective_cv_sklearn(X, y, model_name=name, n_classes = n_classes, SMOTE_on = None, sample_weights = None)

    # Ejecutar búsqueda
    study = optuna.create_study(direction="maximize")
    study.optimize(objective.objective, n_trials=1)

    # Guardar resultados
    best = objective.get_results()
    all_results[name] = {
        'metrics' : best['metrics'],
        "best_f1": best["mean_f1"],
        "params": best["params"],
        "model": best["model"]  # esto requiere deepcopy en el objective
    }
print(all_results)