import numpy as np
import pandas as pd
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess.translate import gen_text_for_embedding, final_clean

def gen_dataset(codes_vrid, df):
    #Selección unicamente de elementos de df que se encuentren en codes_vrid
    df = df[df["Código VRID"].isin(codes_vrid)].copy()

    #Creación de index en función de orden de los datos
    df['idx'] = np.arange(0, df.shape[0])

    #Generación de datasets
    X = df["text_for_embedding_translated"].to_list()
    y = df["Interdisciplinario"].to_list()
    return X, y, df

def gen_dataset_select_cols(codes_vrid, df, cols):
    #Selección unicamente de elementos de df que se encuentren en codes_vrid
    df = df[df["Código VRID"].isin(codes_vrid)].copy()

    #Creación de index en función de orden de los datos
    df['idx'] = np.arange(0, df.shape[0])

    #Generación de datasets
    for col in cols:
        df[col] = df[col].apply(final_clean)

    df=gen_text_for_embedding(df, cols)
    X = df["text_for_embedding_translated"].to_list()
    y = df["Interdisciplinario"].to_list()
    return X, y, df

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

class CvCustom():
    def __init__(self, df_decode, n_splits = None):
        #Dict codes
        self.df_decode=df_decode
        #Lectura de index de separacion de conjuntos train/test
        path = "/tmp/data"
        filepath=os.path.join(path, "train_test_ids_3folds.json")
        with open(filepath, "r", encoding="utf-8") as f:
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