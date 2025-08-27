import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

#Creacion de vectores TFID
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar recursos necesarios (solo la primera vez)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

def preprocess_text_for_TFID(texts):
    """
    Preprocesa textos en inglés para TF-IDF:
    - Minúsculas
    - Tokenización con regex
    - Eliminación de stopwords
    - Lematización
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    processed_texts = []

    for text in texts:
        # Minúsculas
        text = text.lower()

        # Tokenización: mantener solo palabras (a-z)
        tokens = re.findall(r"\b[a-z]+\b", text)

        # Eliminar stopwords y lematizar
        tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens if token not in stop_words
        ]

        processed_texts.append(" ".join(tokens))
    
    return processed_texts

def code_to_idx(list_codes, df):
    """
    Retorna los índices de las filas del DataFrame df 
    en las que la columna 'Código VRID' coincide con 
    alguno de los códigos en list_codes.

    Parámetros
    ----------
    list_codes : list
        Lista de códigos a buscar.
    df : pandas.DataFrame
        DataFrame que contiene la columna 'Código VRID'.

    Retorna
    -------
    list
        Lista con los índices del DataFrame correspondientes a los códigos encontrados.
    """
    mask = df["Código VRID"].isin(list_codes)
    return df.index[mask].tolist()

def gen_TFID_dataset(codes_vrid, df):
    #Selección unicamente de elementos de df que se encuentren en codes_vrid
    df = df[df["Código VRID"].isin(codes_vrid)].copy()

    #Creación de index en función de orden de los datos
    df['idx'] = np.arange(0, df.shape[0])

    #Generación de datasets
    X = df["text_for_embedding_translated"]
    y = df["Interdisciplinario"]
    return X, y, df

def gen_TFID_vectors(X_train, X_test):
    
    #Lemantización y eliminación de stopwords
    X_train = preprocess_text_for_TFID(X_train)
    X_test = preprocess_text_for_TFID(X_test)

    # Crear el vectorizador
    vectorizer = TfidfVectorizer()

    # Ajustar y transformar los documentos
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test
 