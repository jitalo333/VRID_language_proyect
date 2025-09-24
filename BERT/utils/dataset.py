import numpy as np
import pandas as pd
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess.translate import gen_text_for_embedding, final_clean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import json

def binarize_labels(y_train, y_test, positive_labels):
    """
    Binariza los vectores de labels según una lista de condiciones positivas.

    Parámetros:
        y_train (array-like): etiquetas de entrenamiento
        y_test (array-like): etiquetas de prueba
        positive_labels (list): lista de valores que deben considerarse como 1

    Retorna:
        y_train_bin, y_test_bin (arrays numpy con 0 y 1)
    """

    # Normalizar entradas a string y quitar espacios
    y_train = np.char.strip(np.array(y_train).astype(str))
    y_test = np.char.strip(np.array(y_test).astype(str))

    # Crear condición general
    cond_train = np.isin(y_train, positive_labels)
    cond_test = np.isin(y_test, positive_labels)

    # Aplicar binarización
    y_train_bin = np.where(cond_train, 1, 0).astype(int)
    y_test_bin = np.where(cond_test, 1, 0).astype(int)

    return y_train_bin, y_test_bin


def gen_dataset(codes_vrid, df):
    #Selección unicamente de elementos de df que se encuentren en codes_vrid
    df = df[df["Código VRID"].isin(codes_vrid)].copy()

    #Creación de index en función de orden de los datos
    df['idx'] = np.arange(0, df.shape[0])

    #Generación de datasets
    X = df["text_for_embedding_translated"].to_list()
    y = df["Interdisciplinario"].to_list()
    return X, y, df

def gen_dataset_select_cols(codes_vrid, df, cols, element_names=None, test_col="Interdisciplinario"):
    #Selección unicamente de elementos de df que se encuentren en codes_vrid
    df = df[df["Código VRID"].isin(codes_vrid)].copy()

    #Creación de index en función de orden de los datos
    df['idx'] = np.arange(0, df.shape[0])

    #Generación de datasets
    for col in cols:
        df[col] = df[col].apply(final_clean)

    df=gen_text_for_embedding(df, cols, element_names=element_names)
    X = df["text_for_embedding_translated"].to_list()
    y = df[test_col].to_list()
    return X, y, df

def gen_dataset_select_cols_all_dataset(df, cols, element_names=None):
    #Generación de datasets
    for col in cols:
        df[col] = df[col].apply(final_clean)

    df=gen_text_for_embedding(df, cols, element_names=element_names)
    X = df["text_for_embedding_translated"].to_list()
    
    return X

def decoder_vrid(fold_codes, df_decode):
    """
    Decodifica fold_codes usando df_decode.
    fold_codes : lista o array con índices (ej. [0, 2, 5])
    df_decode  : DataFrame con columnas ['idx', 'Código VRID']

    Devuelve un numpy.array con los códigos VRID correspondientes.
    """
    # Crear un diccionario {Código VRID: código}
    mapping = df_decode.set_index("Código VRID")["idx"].to_dict()

    # Mapear los fold_codes a códigos (ignora los que no existan en mapping)
    decoded = [mapping[c] for c in fold_codes if c in mapping]

    return np.array(decoded)

def to_serializable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj

# Función utilizada para seleccionar columna en base a la cual se van a dividir los datos
def get_dataset_to_split(df, feat_col, drop_duplicates_col = "Código VRID"):
    """
    Filtra y selecciona datos de interés desde un DataFrame.

    Inputs:
    - df (pd.DataFrame): DataFrame de entrada con varias columnas.
    - feat_col (str): Nombre de la columna con la característica de interés.
    - drop_duplicates_col (str, opcional): Columna usada para eliminar duplicados
      (por defecto "Código VRID").

    Outputs:
    - pd.DataFrame: DataFrame reducido que contiene solo dos columnas:
      [drop_duplicates_col, feat_col], sin duplicados y sin valores NaN en feat_col.
    """
    df = df[df[feat_col].notna()].drop_duplicates(drop_duplicates_col)
    df = df[[drop_duplicates_col, feat_col]]
    return df

#Funcion  de división de datos
def split_dataset(filepath, ids, labels):
    """
    Divide los datos en train/test, genera folds de validación y guarda índices en JSON.

    Inputs:
    - filepath (str): Ruta donde se guardará el archivo JSON con los índices.
    - ids (array-like): Identificadores de las instancias (ej: códigos VRID).
    - labels (array-like): Etiquetas de las instancias (categorías o clases).

    Outputs:
    - dict: Diccionario con:
        {
            "Test": array de IDs para el conjunto de test,
            "kfolds": lista de arrays de IDs para cada fold de validación
        }
      Además, guarda este diccionario en disco en formato JSON.
    """
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    idx_train, idx_test, y_train, y_test = train_test_split(
        ids,
        labels,
        test_size=0.2,
        random_state=7,
        stratify=labels
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    folds = []
    for _, val_pos in skf.split(idx_train, y_train):
        val_ids = idx_train[val_pos]
        folds.append(val_ids)

    print("Test size:", len(idx_test))
    print("Fold 0 - Val size:", len(folds[0]))

    dataset_index = {
        "Test": idx_test,
        "kfolds": folds
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset_index, f, default=to_serializable, indent=2, ensure_ascii=False)
        print("Archivo guardado exitosamente en", filepath)
    except Exception as e:
        print("Error al guardar el archivo:", e)

class CvCustom():
    def __init__(self, df_decode, path, n_splits = None):
        #Dict codes
        self.df_decode=df_decode
        #Lectura de index de separacion de conjuntos train/test
        with open(path, "r", encoding="utf-8") as f:
            dataset_index = json.load(f)
        folds_codes = dataset_index["kfolds"]
        self.n_splits=len(folds_codes)
        #Define index for kfolds
        self.kfolds = []
        for i in range(self.n_splits):
            self.kfolds.append(decoder_vrid(folds_codes[i], self.df_decode))
            
        #Save al idx
        self.all_idx = np.array([i for fold in self.kfolds for i in fold])
        print(self.all_idx.shape)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        for i in range(self.n_splits):
            test_idx = self.kfolds[i]
            train_idx = np.setdiff1d(self.all_idx, test_idx) 
            yield train_idx, test_idx

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item