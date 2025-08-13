from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer
import torch
import langid
import re
import pandas as pd

class translator():
    """
    Clase para traducir texto del español al inglés utilizando un modelo y tokenizer de Hugging Face.
    Incluye detección de idioma, segmentación en fragmentos y unión de la traducción final.
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

    def translate_esp_en(self, text_to_split):
        """
        Traduce un texto del español al inglés dividiéndolo en fragmentos y recomponiendo el resultado.
        Args:
            text_to_split (str): Texto en español. 
        Returns:
            str: Texto traducido al inglés.
        """
        #Split text
        texts = self.split_text(text_to_split)
        # Translate
        translated_chunks = []
        for chunk in texts:
            encoded = self.tokenizer(chunk.page_content, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                out_ids = self.model.generate(
                    **encoded,
                    num_beams=4,
                    max_new_tokens=self.max_input_tokens,   # evita salidas cortas
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            translated = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
            translated_chunks.append(translated)
        # Traducción final unida
        final_translation = "\n\n".join(translated_chunks)
        return final_translation

    # Detección y traducción
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
    return text.strip()

def gen_text_for_embedding(df, cols):
    """
    df   : DataFrame de entrada
    cols : lista de nombres de columnas a procesar y concatenar
    """
    df = df.copy()
    # Aplica limpieza a cada columna especificada
    for col in cols:
        df[col] = df[col].apply(final_clean)
    # Concatena las columnas limpias en una nueva columna
    df["text_for_embedding_translated"] = df[cols].agg(" ".join, axis=1)
    return df


"""
# Ejemplo de uso
text_to_split = "hola mundo. Este es un texto de prueba para traducir al inglés. Espero que funcione bien."

model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
translator = translator(model, tokenizer)
final_translation=translator.translate_esp_en(text_to_split)
print(final_translation)
"""