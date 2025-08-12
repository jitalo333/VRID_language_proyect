from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer
import torch
import langid

class translator():
    def __init__(self, model, tokenizer, max_input_tokens=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Cargar modelo y tokenizer
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.max_input_tokens = max_input_tokens
    
    def split_text(self, text_to_split):  
        # Splitter basado en el tokenizador de Helsinki (cuenta tokens reales)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.max_input_tokens,       # el encoder de Marian suele aceptar hasta ~512 tokens
            chunk_overlap=0,
            separators=["\n\n", ".", ",", " "]
        )

        texts = text_splitter.create_documents([text_to_split])
        return texts
        
    def translate_esp_en(self, text_to_split):
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
        lang, _ = langid.classify(text)
        if lang == 'es':
            return self.translate_esp_en(text)
        
        return text


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