from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer
import torch
import langid
import re
import pandas as pd
from tqdm import tqdm
import torch

def gen_ids(num, list):
    """
    Repite el ID tantas veces como elementos haya en la lista.
    Args:
        num (int): ID a repetir.
        list (list): Lista cuyos elementos determinan cuántas veces repetir el ID.
    Returns:
        list: Lista con el ID repetido.
    """
    return [num] * len(list)
    
def detect_language(texts):
    """
    Detecta si los textos están en español.
    Args:
        texts (list): Lista de textos a evaluar.
    Returns:
        list: Lista de booleanos indicando si cada texto está en español.
    """
    langs =[]
    for text in texts:
        lang, _ = langid.classify(text)
        if lang == 'es':
            langs.append(True)
        else:
            langs.append(False)
    return langs

class translator():
    """
    Clase para traducir texto del español al inglés utilizando un modelo y tokenizer de Hugging Face.
    Incluye detección de idioma, segmentación en fragmentos y unión de la traducción final.
    La traducción puede realizarse de forma individual, utilizando detect_and_translate(), 
    o bien procesar una lista de textos en paralelo mediante translate_parallel().

    """
    def __init__(self, model, tokenizer, max_input_tokens=512):
        """
        Inicializa el traductor cargando el modelo y tokenizer en GPU si está disponible.
        Args:
            model: Modelo de traducción.
            tokenizer: Tokenizer asociado al modelo.
            max_input_tokens (int): Máximo de tokens por fragmento.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        # Cargar modelo y tokenizer
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.max_input_tokens = max_input_tokens

    def split_text(self, text_to_split):
        """
        Divide un texto en fragmentos manejables según el límite de tokens del modelo.
        Args:
            text_to_split (str): Texto original a dividir.
        Returns:
            list: Lista de fragmentos como objetos Document.
        """
        # Splitter basado en el tokenizador de Helsinki (cuenta tokens reales)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.max_input_tokens,
            chunk_overlap=0,
            separators=["\n\n", ".", ",", " "] #Jerarquía de separadores
        )

        texts = text_splitter.create_documents([text_to_split])
        return texts

    def translate_esp_en(self, text_to_split, batch_size=16):
        texts = self.split_text(text_to_split)
        translated_chunks = []

        for i in range(0, len(texts), batch_size):
            batch = [t.page_content for t in texts[i:i+batch_size]]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                out_ids = self.model.generate(
                    **encoded,
                    num_beams=4,
                    max_new_tokens=self.max_input_tokens,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            translated_batch = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            translated_chunks.extend(translated_batch)

        return "\n\n".join(translated_chunks)

    # Detección y traducción de texto. 
    def detect_and_translate(self, text):
        """
        Detecta el idioma del texto y lo traduce si está en español.
        Args:
            text (str): Texto de entrada.
        Returns:
            str: Texto traducido o el mismo texto si no es español.
        """
        lang, _ = langid.classify(text)
        if lang == 'es':
            return self.translate_esp_en(text)
        
        return text

    def translate_parallel(self, texts, batch_size=16):
        """
        Traduce en paralelo una lista de textos mezclados en español e inglés.

        Inputs:
            texts (list[str]): Lista de textos a traducir.

        Outputs:
            list[str]: Lista de textos donde los que estaban en español fueron traducidos
                    y los que estaban en otros idiomas se mantienen igual.

        Proceso:
            1. Detecta qué textos están en español.
            2. Divide los textos largos en fragmentos (para no superar límite de tokens).
            3. Traduce por lotes con el modelo de traducción.
            4. Reconstruye los textos traducidos completos.
            5. Une los textos traducidos con los originales en otros idiomas, manteniendo orden.
        """

        # Crear lista de IDs únicos para no perder el orden
        ids = list(range(len(texts)))

        # Detectar idioma de cada texto (True = español, False = otro idioma)
        langs = detect_language(texts)
   

        # Construir dataframe base
        df = pd.DataFrame({'id': ids, 'is_spanish': langs, 'text': texts})

        # Separar en textos español e inglés
        df_spanish = df[df['is_spanish']].copy()
        df_other   = df[~df['is_spanish']].copy()

        # Dividir textos españoles en fragmentos manejables
        df_spanish['text'] = df_spanish['text'].apply(self.split_text)

        # Generar lista expandida de (id, fragmento)
        id_loc, texts_for_batch = [], []
        for _, row in df_spanish.iterrows():
            id_loc.extend(gen_ids(row['id'], row['text']))   # genera ID por fragmento
            texts_for_batch.extend(row['text'])              # agrega fragmentos

        # === Traducción por lotes ===
        translated_chunks = []
        
        for i in tqdm(range(0, len(texts_for_batch), batch_size), desc="Traduciendo", unit="batch"):
            batch = [t.page_content for t in texts_for_batch[i:i+batch_size]]
            
            # Tokenizar batch
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generación de traducción
            with torch.inference_mode():
                out_ids = self.model.generate(
                    **encoded,
                    num_beams=4,
                    max_new_tokens=self.max_input_tokens,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            # Decodificar traducciones y acumular
            translated_batch = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            translated_chunks.extend(translated_batch)

        # === Reconstruir textos ===
        df_translated = pd.DataFrame({'id': id_loc, 'text': translated_chunks})
        df_translated = (
            df_translated.groupby('id')['text']
            .apply(lambda x: ' '.join(x))      # unir fragmentos del mismo texto
            .reset_index()
        )

        # Unir con textos originales en otros idiomas y ordenar
        df_final = pd.concat(
            [df_translated, df_other.drop(columns=['is_spanish'])],
            ignore_index=True
        ).sort_values(by="id").reset_index(drop=True)

        return df_final["text"].to_list()
    
def gen_text_for_embedding(df, cols, element_names=None, sep=" "):
    """
    df            : DataFrame de entrada
    cols          : lista de nombres de columnas a procesar y concatenar
    element_names : lista de etiquetas para cada columna (misma longitud que cols)
    sep           : separador entre pares clave-valor (default = " ")
    """
    if element_names is None:
        df["text_for_embedding_translated"] = df[cols].agg(sep.join, axis=1)
    else:
        if len(element_names) != len(cols):
            raise ValueError("element_names debe tener la misma longitud que cols")

        df["text_for_embedding_translated"] = df[cols].agg(
            lambda row: sep.join(
                f"{name}: {row[col]}" for col, name in zip(cols, element_names)
            ),
            axis=1
        )
    return df

def final_clean(text):
    """
    Limpia un texto eliminando saltos de línea, tabs y espacios múltiples.
    Args:
        text (str): Texto de entrada.
    Returns:
        str: Texto limpio y sin espacios innecesarios.
    """
    if not isinstance(text, str):
        return ""
    # Reemplaza saltos de línea y tabs por un espacio
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Colapsa espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    #Lleva todo a minúscula
    text = text.lower()
    
    return text.strip()

def translate_OCDE_features(df, col, df_translations):
    """
    Translate the values of a column in df using a dictionary 
    generated from df_translations (esp, traduccion), ignoring case sensitivity.

    Inputs:
    - df (pd.DataFrame): DataFrame with the column to translate.
    - col (str): Name of the column in df that contains the Spanish terms.
    - df_translations (pd.DataFrame): DataFrame with columns ["esp", "traduccion"].

    Output:
    - pd.Series: Column translated into English (keeps original if no translation is found).
    - dict: Dictionary used for translation (with lowercased keys).
    """
    # Crear un diccionario {español: inglés}, asegurando strings y saltando NaN
    dictionary = {
    " ".join(str(k).split()): v
    for k, v in zip(df_translations["esp"], df_translations["traduccion"])
    if pd.notna(k) and pd.notna(v)
    }
    #Aplicar limpieza a columna que se va a traducir
    df[col] = df[col].map(final_clean)
    # Convertir a string, minúsculas y mapear traducciones
    translated = df[col].astype(str).str.lower().map(dictionary)

    # Donde no hay traducción, devolver el valor original
    translated = translated.fillna(df[col])

    return translated
