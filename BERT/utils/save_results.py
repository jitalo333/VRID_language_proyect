import git
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import tempfile
import matplotlib.pyplot as plt
import json
import io
from PIL import Image
import pickle
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline


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

def register_confusion_matrix(df_cm, class_labels=None):
    """
    Genera una imagen de la matriz de confusión normalizada.

    Parámetros:
        df_cm (pd.DataFrame): matriz de confusión con índices y columnas como clases
        class_labels (list, opcional): etiquetas de clase a mostrar. 
                                       Si None, usa las del DataFrame.

    Retorna:
        PIL.Image con la matriz de confusión
    """
    cm = df_cm.to_numpy().astype(float)
    # Normalizar por filas (cada fila suma 1)
    cm = cm / cm.sum(axis=1, keepdims=True)

    # Etiquetas de clases
    if class_labels is None:
        class_labels = df_cm.columns.tolist()

    # Crear la figura
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title("Matriz de Confusión")
    fig.colorbar(im, ax=ax)

    # Agregar valores dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.3f}",
                    ha="center", va="center", color="black")

    # Configurar ticks con etiquetas correctas
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    plt.tight_layout()

    # Guardar en memoria como PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf)
    plt.close(fig)

    return img

def model_to_pipeline(vectorizer, models_dicc):
    models_dicc_pipeline = {}
    for name, model in models_dicc.items():
        pip = Pipeline([("TF-IDF", vectorizer), ("model", model)])
        models_dicc_pipeline[name] = pip

    return models_dicc_pipeline

def save_models_and_metrics(path, results_val, models_dicc, df_test, y_test, results_test, preds, save_preds=None, mode_classification="binary"):
    
    # Obtener commit actual
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    #Diccionario donde se guardará todo
    save_dict = {}

    #Make metrics folders
    path_metrics = os.path.join(path, "metrics")
    os.makedirs(path_metrics, exist_ok=True)

    #Make models folders
    path_models = os.path.join(path, "models")
    os.makedirs(path_models, exist_ok=True)

    for model_name, metrics in results_val.items():
       
        model = models_dicc[model_name]
        print(f"📝 Registrando modelo: {model_name}")

        # Hiperparámetros
        try:
            params = model.get_params()
            save_path = os.path.join(path_models, f"{model_name}.joblib")
            joblib.dump(model, save_path)

        except:
            print(f"⚠️ No se pudieron loggear los hiperparámetros para {model_name}")

        # Métricas de validación
        for k, v in metrics.items():
            save_dict[f"val_{k}"] = v

        #Predicciones y resultados de test
        
        #Guardar predicciones
        if save_preds is not None:
            df_test["y_true"]=y_test
            df_test["preds"]=preds
            df_test = df_test[["Código VRID", "y_true", "preds"]]
            #Save as csv
            save_path = os.path.join(path_metrics, f"preds_{model_name}.csv")
            df_test.to_csv(save_path, index=False, encoding="utf-8-sig")

        # Guardar métricas de test
        for k, v in results_test.items():
            if k.startswith("cm"):
                # Guardar confusion matrix (o similar) como artefacto
                # Guardar como CSV temporal
                v = pd.DataFrame(v)
                save_path = os.path.join(path_metrics, f"cm_{model_name}.csv")
                v.to_csv(save_path, index=False, encoding="utf-8-sig")
                #Guardar imagen
                cm_img = register_confusion_matrix(v)
                save_path = os.path.join(path_metrics, f"cm_{model_name}.png")
                cm_img.save(save_path, format="PNG")
                
            else:
                # Guardar métrica numérica
                save_dict[f"test_{k}"]  = v
        
        #Guardar commit de git
        save_dict["git_commit"] = commit_hash

        #Guardar diccionario con métricas
        save_path = os.path.join(path_metrics, f"{model_name}.json")
        # Guardar en JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)

def load_model(path):
    inference_model = joblib.load(path)
    return inference_model
