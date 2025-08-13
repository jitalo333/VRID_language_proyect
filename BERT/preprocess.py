import pandas as pd
import re
import unicodedata

def get_expresions_to_delete():
  """
  Devuelve una lista de patrones regex para eliminar 
  instrucciones o textos no deseados en el resumen.
  """
  delete = [
    r"resumen del proyecto\s*\(1\s*p[aá]gina\)",
    r"debe ser suficientemente informativo.*?proyecto",
    r"problema que se abordar[áa],\s*objetivos,\s*metodolog[ií]a y resultados que se esperan[\s\S]*?de evaluadores",
    r"problema que se abordar[áa],\s*objetivos,\s*metodolog[ií]a y resultados que se esperan[\s\S]*?investigaci[oó]n",
    r"debe considerarse que un resumen bien formulado facilita.*?evaluadores",
    r"DESCRIBE THE MAIN ISSUES TO BE ADDRESSED[\s\S]*?EXPECTED RESULTS\.",
    r"THE MAXIMUM LENGTH FOR THIS SECTION[\s\S]*?SIMILAR\).",
    r"AVOID INCLUDING IN THIS SECTION INFORMATION[\s\S]*?BACKGROUNDS\."
    ]
  return delete

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Normalizar caracteres unicode (acentos, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Estandarización de carácteres de espacio
    text = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200F\u202F\u205F\u3000\uFEFF]', ' ', text)
    text = re.sub(r'\_x000D_', ' ', text)

    # Eliminar referencias tipo [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Eliminar números tipo "1. Introducción"
    text = re.sub(r'\d+\.\s*[A-Za-zÁÉÍÓÚáéíóúñÑ]+', '', text)

    # Eliminar caracteres no alfanuméricos (excepto puntuación básica)
    #text = re.sub(r'[^\w\s.,;:()\[\]¿?!¡%\-\\n]', '', text)

    # Eliminar múltiples espacios
    text = re.sub(r'[ \t]+', ' ', text)

    # Eliminar instrucciones comunes del formulario
    delete = get_expresions_to_delete()
    for prhase in delete:
        text = re.sub(prhase, '', text, flags=re.IGNORECASE | re.DOTALL)

    #busca cualquier secuencia de 3 o más saltos de línea consecutivos.
    text = re.sub(r'\n{3,}', '\n', text)

    return text.strip().lower()

#Esta función aún no funciona completamente bien, así que no está en el Pipeline
def expand_acronyms(text):
    acronyms = {}
    # Encontrar definiciones de acrónimos tipo "Texto largo (ACR)"
    pattern = re.compile(r'\b([A-Z][A-Za-z0-9&.\s]+?)\s*\(\s*([A-Z]{2,})\s*\)')
    for match in pattern.finditer(text):
        long_form, short_form = match.groups()
        acronyms[short_form] = long_form.strip()

    # Eliminar la definición original dejando solo la forma larga
    text = pattern.sub(lambda m: m.group(1), text)

    # Reemplazar todas las apariciones del acrónimo por la forma larga
    if acronyms:
        acronym_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, acronyms.keys())) + r')\b')
        text = acronym_pattern.sub(lambda m: acronyms[m.group(0)], text)

    return text



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