import pandas as pd
import re
import unicodedata

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Normalizar caracteres unicode (acentos, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Reemplazar saltos de línea reales por el string literal '\n'
    text = text.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')

    # Reemplazar todos los caracteres de espacio Unicode raros por un espacio común
    text = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200F\u202F\u205F\u3000\uFEFF]', ' ', text)

    # Eliminar referencias tipo [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Eliminar números tipo "1. Introducción"
    text = re.sub(r'\d+\.\s*[A-Za-zÁÉÍÓÚáéíóúñÑ]+', '', text)

    # Eliminar caracteres no alfanuméricos (excepto puntuación básica)
    text = re.sub(r'[^\w\s.,;:()\[\]¿?!¡%\-\\n]', '', text)

    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)

    # Eliminar instrucciones comunes del formulario 
    delete = [
        r"resumen del proyecto\s*\(1\s*p[aá]gina\)",
        r"debe ser suficientemente informativo y claro.*?proyecto",
        r"problema que se abordar[áa], objetivos, metodolog[ií]a y resultados que se esperan.*?investigaci[oó]n",
        r"debe considerarse que un resumen bien formulado facilita.*?evaluadores"
    ]
    #agregar: DESCRIBE THE MAIN ISSUES TO BE ADDRESSED: OBJECTIVES, METHODOLOGY AND EXPECTED RESULTS. THE MAXIMUM
    #LENGTH FOR THIS SECTION IS 1 PAGE (USE LETTER SIZE FORMAT, VERDANA FONT SIZE 10 OR SIMILAR).
    for prhase in delete:
        text = re.sub(prhase, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


def preprocess_record(title, abstract, keywords, max_keywords=20):
    clean_abs = clean_text(abstract)
    title_clean = clean_text(title)
    kw_list = [k.strip() for k in keywords.split(';') if k.strip()][:max_keywords]
    kw_list = [clean_text(k) for k in kw_list]

    weighted_parts = [
        (title_clean, 1.0),
        (f"Keywords: {'; '.join(kw_list)}", 1.0) if kw_list else ("", 0),
        (f"Abstract: {clean_abs}", 1.0),
    ]

    parts = [part for part, weight in weighted_parts for _ in range(int(weight))]
    return ". ".join(filter(None, parts)).lower()



"""
################################ Ejemplo de uso ####################################
import os
import pandas as pd
from preprocess import preprocess_record

# 1) Cargar datos
path = "/content/drive/MyDrive/VRID_NLP/code/VRID_proyect/"
filePATH = os.path.join(path, "data_concatenada.xlsx")
df = pd.read_excel(filePATH,
                   usecols=["Código VRID", "Título", "Resumen", "Keywords", "Interdisciplinario", "Transdisciplinario"]) \
       .fillna("")
cols = ["Título", "Resumen", "Keywords"]
df[cols] = df[cols].applymap(lambda x: "" if str(x).strip().upper() == "DESCONOCIDO" else str(x).strip())
df["text_for_embedding"] = df.apply(
    lambda r: preprocess_record(r["Título"], r["Resumen"], r["Keywords"]),
    axis=1
)
df.to_excel("peso1.xlsx")

"""