import os
import json
import numpy as np

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
