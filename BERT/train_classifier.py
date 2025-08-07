############### Este es un pipeline que hace lo mismo que train_classifier.ipynb, pero en un script .py ###############

import pandas as pd
import os
from preprocess import preprocess_record
from translate import translator
from encoder import embed_texts, prepare_data
import torch
import argparse

def main(path):
    # Parámetros generales
    BASE_MODEL = "allenai/specter2_base"
    ADAPTER_NAME = "allenai/specter2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Cargar datos
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
    df=df.iloc[0:10]

    # 3) Traducir los textos al inglés
    trans = translator()
    df["text_for_embedding_translated"] = df.apply(
        lambda r: trans.detect_and_translate(r.get("text_for_embedding", "")),
        axis=1
    )
    df.to_excel("data_translated.xlsx", index=False)


    # 4) Preparar textos y etiquetas
    texts, labels = prepare_data(df)

    # 4) Calcular embeddings
    # Parámetros modelo
    BASE_MODEL = "allenai/specter2_base"
    ADAPTER_NAME = "allenai/specter2"
    emb_texts = embed_texts(texts, BASE_MODEL, ADAPTER_NAME)

    # 6) Guardar embeddings y etiquetas
    df_dataset = pd.DataFrame(columns=["Código VRID", "labels", "embedings"])
    df_dataset["Código VRID"] = df.loc[labels.index, "Código VRID"]
    df_dataset["labels"] = labels
    df_dataset["embedings"] = emb_texts.tolist()
    df_dataset.to_excel("dataset_embed_translated.xlsx", index=False)


if __name__ == "__main__":
    # Definir la ruta del directorio actual con parser

    parser = argparse.ArgumentParser(description="Pipeline para entrenar un clasificador.")
    parser.add_argument("--path", type=str, default=".", help="Ruta del directorio donde se encuentran los datos.")
    args = parser.parse_args()
    main(args.path)

    # Mensaje de finalización
    print("Pipeline completado exitosamente.")