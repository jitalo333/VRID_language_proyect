#Confusion matrix
import io
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import numpy as np
import mlflow
import os
import git
import tempfile
import matplotlib.pyplot as plt


def register_confusion_matrix(df_cm):
    cm = df_cm.to_numpy().astype(float)
    # Normalizar por filas (cada fila suma 1)
    cm = cm / cm.sum(axis=1, keepdims=True)
    # Crear la figura
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusión")
    fig.colorbar(im, ax=ax)

    # Agregar valores dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.3f}",
                    ha="center", va="center", color="black")

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.tight_layout()

    # Guardar en memoria como PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Convertir a PIL.Image
    img = Image.open(buf)

    # Importante: cerrar figura para que no se muestre ni ocupe memoria
    plt.close(fig)

    return img

def metrics_lang(y, preds, lang_es):
    #Conversión en array
    y = np.array(y)
    preds = np.array(preds)
    lang_es = np.array(lang_es)

    # Seleccionar por máscara booleana
    y_es = y[lang_es] #Data originalmente en español
    preds_es = preds[lang_es]

    y_en = y[~lang_es] #Data originalmente en inglés
    preds_en = preds[~lang_es]
    
    #Computo de métricas
    f1_es = f1_score(y_es, preds_es, average="weighted")
    f1_en = f1_score(y_en, preds_en, average="weighted")
    cm_es = confusion_matrix(y_es, preds_es)
    cm_en = confusion_matrix(y_en, preds_en)

    return f1_es, f1_en, cm_es, cm_en

def eval_model(best_model, X_test, y_test, lang_es=None, mode = "binary"):
    results = {}
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    #Métricas generales
    results = {
        'accuracy': accuracy_score(y_test, preds),
        'f1_macro': f1_score(y_test, preds, zero_division=0, average="macro"),
        'cm': cm,
    }
    # Métricas adicionales para clasificación binaria
    if mode == "binary":
        results.update({
            'precision': precision_score(y_test, preds, zero_division=0),
                'recall': recall_score(y_test, preds, zero_division=0)
            })
    # Métricas por idioma
    if lang_es is not None:
        f1_es, f1_en, cm_es, cm_en = metrics_lang(y_test, preds, lang_es)
        results.update({
            'f1_es': f1_es,
            'f1_en': f1_en,
            'cm_es': cm_es,
            'cm_en': cm_en
        })
    return results, preds

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

def log_artifact_generic(k, v):
    #Guardar artefactos csv y json
    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(v, dict):
            # Guardar como JSON
            fname = os.path.join(tmpdir, f"{k}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(v, f, indent=4, ensure_ascii=False)

        elif isinstance(v, pd.DataFrame):
            # Guardar como CSV
            fname = os.path.join(tmpdir, f"{k}.csv")
            v.to_csv(fname, index=False)

        elif isinstance(v, str) and os.path.exists(v):
            # Si ya es una ruta válida
            fname = v
        
        elif isinstance(v, Image.Image):
            fname = os.path.join(tmpdir, f"{k}.png")
            v.save(fname, format="PNG")

        else:
            # Si no sabes qué es, lo guardamos como string plano
            fname = os.path.join(tmpdir, f"{k}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(str(v))

        # Log en MLflow
        mlflow.log_artifact(fname, artifact_path=k)

def mlflow_ckeckpoint(exp_info, results_val, models_dicc, X_test, y_test, df_test, save_preds=None, lang_es=None, extra_parms=None, extra_artifacts = None, mode="server", mode_classification="binary"):
    
    if mode == "server":
        # Set backend store
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri)) 
    
    elif mode == "local": 
        # Set backend store
        mlflow.set_tracking_uri(exp_info["tracking_path"])
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri)) 

        # Verificar si existe experimento, si no crearlo
        experiment = mlflow.get_experiment_by_name(exp_info["exp_name"])

        if experiment is None:
            exp_id = mlflow.create_experiment(
                exp_info["exp_name"],
                artifact_location=exp_info["artifact_path"]
            )
            print(f"Experimento creado con ID: {exp_id}")
        else:
            exp_id = experiment.experiment_id
            print(f"Experimento ya existe con ID: {exp_id}")
    
    else: 
        print("Especificar modo de almacenamiento")
        return 0

    # Define el experimento (lo crea si no existe)
    mlflow.set_experiment(exp_info["exp_name"])
    
    # Obtener commit actual
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    for model_name, metrics in results_val.items():
        model = models_dicc[model_name]

        with mlflow.start_run(run_name=model_name):
            print(f"📝 Registrando modelo en MLflow: {model_name}")

            # Hiperparámetros
            try:
                mlflow.log_params(model.get_params())
            except:
                print(f"⚠️ No se pudieron loggear los hiperparámetros para {model_name}")

            #Parámetros adicionales, se pueden añadir con un diccionario
            if extra_parms is not None:
                for k, v in extra_parms.items():
                    mlflow.log_param(k, v)

            #Artefactos adicionales, pueden ser json, csv, txt, dataframe
            if extra_artifacts is not None:
                for k, v in extra_artifacts.items():
                    log_artifact_generic(k, v)

            # Métricas de validación
            for k, v in metrics.items():
                safe_log_metric(f"val_{k}", v)

            #Predicciones y resultados de test
            results_test, preds = eval_model(model, X_test, y_test, lang_es, mode = mode_classification)

            # Generar y guardar matriz de confusión
            #cm_img = register_confusion_matrix(y_test, preds)
            #log_artifact_generic("cm_normalized_img", cm_img)

            #Guardar predicciones
            if save_preds is not None:
                df_test["y_true"]=y_test
                df_test["preds"]=preds
                log_artifact_generic("df_test", df_test)

            # Guardar métricas de test
            for k, v in results_test.items():
                if k.startswith("cm"):
                    # Guardar confusion matrix (o similar) como artefacto
                    # Guardar como CSV temporal
                    v = pd.DataFrame(v)
                    log_artifact_generic(k, v)
                    #Guardar imagen
                    cm_img = register_confusion_matrix(v)
                    log_artifact_generic(f"{k}_img", cm_img)
                    
                else:
                    # Guardar métrica numérica
                    safe_log_metric(f"test_{k}", v)
            
            #Guardar commit de git
            mlflow.log_param("git_commit", commit_hash)
                    
            # Guardar modelo
            mlflow.sklearn.log_model(model, artifact_path = "model", input_example=X_test[:5])
 