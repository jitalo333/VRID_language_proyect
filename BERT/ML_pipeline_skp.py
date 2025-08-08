from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

import copy
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.decomposition import PCA

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import numpy as np

import mlflow

def eval_model(best_model, X_test, y_test):
  results = {}
  preds = best_model.predict(X_test)
  cm = confusion_matrix(y_test, preds)
  t_n, f_p, f_n, t_p = cm.ravel()
  results = {
      'accuracy': accuracy_score(y_test, preds),
      'precision': precision_score(y_test, preds, zero_division=0),
      'recall': recall_score(y_test, preds, zero_division=0),
      'f1_score': f1_score(y_test, preds, zero_division=0),
      't_n': t_n,
      'f_p': f_p,
      'f_n': f_n,
      't_p': t_p
  }
  return results

#MLflow logging helper function
def safe_log_metric(name, value):
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            if np.size(value) == 1:
                value = float(np.array(value).item())
            else:
                raise ValueError("Métrica con más de un valor.")
        else:
            value = float(value)
        mlflow.log_metric(name, value)
    except Exception as e:
        print(f"⚠️ No se pudo loggear {name}: {e}")

def mlflow_ckeckpoint(results_val, models_dicc, X_test, y_test, experiment_name):
    # Define el experimento (lo crea si no existe)
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    for model_name, metrics in results_val.items():
        model = models_dicc[model_name]

        with mlflow.start_run(run_name=model_name):
            print(f"Registrando modelo en MLflow: {model_name}")

            # Hiperparámetros
            try:
                mlflow.log_params(model.get_params())
            except:
                print(f"No se pudieron loggear los hiperparámetros para {model_name}")

            # Métricas de validación
            for k, v in metrics.items():
                safe_log_metric(f"val_{k}", v)

            # Métricas de test
            results_test = eval_model(model, X_test, y_test)
            for k, v in results_test.items():
                safe_log_metric(f"test_{k}", v)

            # Guardar modelo
            mlflow.sklearn.log_model(model, name = "model", input_example=X_test[:5])  

#Pipeline helper functions
def get_est_params_dict(keys):
    clf_params_dict = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {
                'solver': Categorical(['lbfgs']),
                'penalty': Categorical(['l2']),
                'C': [0.1, 1.0, 10]
            }
        },
        'RandomForestClassifier': {
            'class': RandomForestClassifier,
            'params': {
                'bootstrap': Categorical([True]),
                'n_estimators': Integer(40, 300),
                'max_depth': Integer(3, 14),
                'min_samples_split': Integer(2, 5),
                'min_samples_leaf': Integer(1, 2),
            }
        },
        'GradientBoostingClassifier': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': Integer(50, 200),
                'learning_rate': Real(0.01, 0.2),
                'max_depth': Integer(3, 7)
            }
        },
        'XGBClassifier': {
            'class': XGBClassifier,
            'params': {
                'n_estimators': Integer(40, 300),
                'learning_rate': Real(0.01, 0.2),
                'max_depth': Integer(3, 14),
                'eval_metric': Categorical(['logloss'])
            }
        },

    }

    # Filtrar y retornar solo los modelos solicitados
    return {key: clf_params_dict[key] for key in keys if key in clf_params_dict}

def setup_model(dicc):
    model_ = dicc['class']()
    #scaler = StandardScaler()
    model = Pipeline([('model', model_)])
    param_grid = {'model__' + param_name: param_value for param_name, param_value in dicc['params'].items()}
    return model, param_grid

def run_BayesSearchCV(model, param_grid, X_train, y_train, n_iter=10, scoring='recall', sample_weight=None):
    from sklearn.exceptions import FitFailedWarning
    import warnings

    bayes_searchCV = BayesSearchCV(
        estimator=model,
        search_spaces=param_grid,
        cv=5,
        scoring=scoring,
        n_iter = n_iter,
        n_jobs=4,
        n_points = 2
    )

    try:
        fit_params = {'model__sample_weight': sample_weight} if sample_weight is not None else {}
        bayes_searchCV.fit(X_train, y_train, **fit_params)
    except TypeError as e:
        print(f"⚠️ Modelo {model.named_steps['model'].__class__.__name__} no acepta sample_weight. Reintentando sin él.")
        bayes_searchCV.fit(X_train, y_train)

    return bayes_searchCV

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUROC y AUPRC requieren probabilidades (y_proba)
    if y_proba is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_proba)
            metrics["auprc"] = average_precision_score(y_true, y_proba)
        except:
            metrics["auroc"] = np.nan
            metrics["auprc"] = np.nan

    return metrics

def get_sample_weight(y_train):
    """
    Devuelve un array de sample weights donde las muestras con y=1
    tienen 10 veces más peso que las muestras con y=0.
    """
    y_train = np.asarray(y_train)
    weights = np.ones_like(y_train, dtype=np.float64)
    weights[y_train == 1] = 10
    return weights

def select_best_model(results_val, models_dicc):
    best_result = 0
    best_model = None
    for model, metrics in results_val.items():
        if metrics['mean_test_score'] > best_result:
            best_result = metrics['mean_test_score']
            best_model = models_dicc[model]
    return best_model

#Pipeline function to run the entire ML pipeline
def run_bayesian_pipeline(est_params_dict, data, labels, n_iter, sample_weight_On = None):

    # creacion de diccionarios para almacenamiento
    results_test = {}
    cm_test = {}
    results_val = {}
    models_dicc = {}

    for model_name, dicc in est_params_dict.items():

        print(model_name)
        model, param_grid = setup_model(dicc)

        #Sample weights strategy
        if sample_weight_On is not None:
          print('Compute sw')
          sample_weight = compute_sample_weight(class_weight='balanced', y=labels)
        else:
          sample_weight = None
        grid_search = run_BayesSearchCV(model, param_grid, data, labels, n_iter = n_iter, scoring = 'recall', sample_weight = sample_weight)


        # Guardar best model
        best_model = grid_search.best_estimator_
        models_dicc[model_name] = copy.deepcopy(best_model)

        mean_val_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
        std_val_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        mean_val_score = np.round(mean_val_score,2)
        std_val_score = np.round(std_val_score,2)

        metrics_val = {
          'mean_test_score': float(np.round(mean_val_score, 2)),
          'std_test_score': float(np.round(std_val_score, 2)
        )}

        results_val[model_name]  = metrics_val


    return results_val, models_dicc


# Ejemplo de uso
# 1. Generar dataset sintético
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    #weights=[0.95],  # 90% clase 0, 10% clase 1
    random_state=42,
    shuffle=True
)

# 2. Elegir modelos a probar
model_keys = [
    #'LogisticRegression',
    #'DecisionTreeClassifier',
    #'RandomForestClassifier',
    #'GradientBoostingClassifier',
    #'XGBClassifier',
    'MLPClassifier',
    #'SVC',
    #'SGDClassifier'
]

# 3. Obtener el diccionario de modelos y parámetros
est_params_dict = get_est_params_dict(model_keys)

print("📊 Comienzo:", Counter(y))
# 4. Ejecutar entrenamiento, validación y test con tus funciones
results_val, models_dicc = run_bayesian_pipeline(est_params_dict, X, y, n_iter=20, sample_weight_On = True)

best_model = select_best_model(results_val, models_dicc)

# 5. Mostrar resultados
print("\n🔍 Validación:")
for model, metrics in results_val.items():
    print(f"{model}: {metrics}")