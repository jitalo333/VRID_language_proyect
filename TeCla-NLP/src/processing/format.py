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

    # Eliminar caracteres no alfanuméricos molestos (excepto puntuación básica)
    text = re.sub(r'[^\w\s.,;:()\[\]¿?!¡%\-\\n]', '', text)

    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)

    # Eliminar instrucciones comunes del formulario (una por una, de forma flexible)
    frases_a_eliminar = [
        r"resumen del proyecto\s*\(1\s*p[aá]gina\)",
        r"debe ser suficientemente informativo y claro.*?proyecto",
        r"problema que se abordar[áa], objetivos, metodolog[ií]a y resultados que se esperan.*?investigaci[oó]n",
        r"debe considerarse que un resumen bien formulado facilita.*?evaluadores"
    ]
    for frase in frases_a_eliminar:
        text = re.sub(frase, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


def preprocess_record(title, abstract, keywords, max_keywords=20):
    clean_abs = clean_text(abstract)
    title_clean = clean_text(title)
    kw_list = [k.strip() for k in keywords.split(';') if k.strip()][:max_keywords]

    weighted_parts = [
        (title_clean, 1.0),
        (f"Keywords: {'; '.join(kw_list)}", 1.0) if kw_list else ("", 0),
        (f"Abstract: {clean_abs}", 1.0),
    ]

    parts = [part for part, weight in weighted_parts for _ in range(int(weight))]
    return ". ".join(filter(None, parts)).lower()