############### Este es un pipeline que hace lo mismo que train_classifier.ipynb, pero en un script .py ###############

import pandas as pd
import os
from preprocess import preprocess_record
from translate import translator
from encoder import embed_texts
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch

# Parámetros generales
BASE_MODEL = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Cargar datos
path = "\data"
filePATH = os.path.join(path, "data_concatenada.xlsx")
df = pd.read_excel(filePATH,
                   usecols=["Código VRID", "Título", "Resumen", "Keywords", "Interdisciplinario", "Transdisciplinario"]) \
       .fillna("")
# 2) Preprocesar los datos
cols = ["Título", "Resumen", "Keywords"]
df[cols] = df[cols].applymap(lambda x: "" if str(x).strip().upper() == "DESCONOCIDO" else str(x).strip())
df["text_for_embedding"] = df.apply(
    lambda r: preprocess_record(r["Título"], r["Resumen"], r["Keywords"]),
    axis=1
)
df.to_excel("data_preprocessed.xlsx", index=False)

# 3) Traducir los textos al inglés
trans = translator()
df["text_for_embedding_translated"] = df.apply(
    lambda r: trans.detect_and_translate(r.get("text_for_embedding", "")),
    axis=1
)
df.to_excel("data_translated.xlsx", index=False)


# 4) Preparar textos y etiquetas
texts = df["text_for_embedding_translated"].tolist()
labels = df["Interdisciplinario"]
#Eliminar labels = indefinido
labels = labels[labels != "INDEFINIDO"]
#label encoder
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = pd.Series(labels, index=df[df["Interdisciplinario"] != "INDEFINIDO"].index)
texts = [texts[i] for i in labels.index]

# 5) Cargar modelo SPECTER2
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoAdapterModel.from_pretrained(BASE_MODEL, trust_remote_code=True)
model.load_adapter(ADAPTER_NAME, source="hf", set_active=True)
model.to(device)
model.eval()
print("Modelo SPECTER2 cargado correctamente.")

# 6) Calcular embeddings
emb_texts = embed_texts(texts)

# 7) Guardar embeddings y etiquetas
df_dataset = pd.DataFrame(columns=["Código VRID", "labels", "embedings"])
df_dataset["Código VRID"] = df.loc[labels.index, "Código VRID"]
df_dataset["labels"] = labels
df_dataset["embedings"] = emb_texts.tolist()
df_dataset.to_excel("dataset_embed_translated.xlsx", index=False)