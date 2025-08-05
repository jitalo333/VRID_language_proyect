import re
import pandas as pd
import numpy as np

# Funciones auxiliares
def clean_abstract(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d+\.\s*[A-Za-zÁÉÍÓÚáéíóú]+', '', text)
    return text.strip()

def preprocess_record(title, abstract, keywords, journal="", max_keywords=10):
    clean_abs = clean_abstract(abstract)
    kw_list = [k.strip() for k in keywords.split(';') if k.strip()][:max_keywords]
    weighted_parts = [
        (title.strip(), 3.0),
        (f"Keywords: {'; '.join(kw_list)}", 2.0) if kw_list else ("", 0),
        (f"Abstract: {clean_abs}", 1.0),
        (f"Published in: {journal.strip()}", 1.5) if journal else ("", 0)
    ]
    parts = [part for part, weight in weighted_parts for _ in range(int(weight))]
    return ". ".join(filter(None, parts))
