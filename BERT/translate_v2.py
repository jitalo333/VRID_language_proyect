import langid
import torch


def _translate_block_bart(text_block, model, tokenizer, device):
    """
    Traduce un bloque de texto del español al inglés usando MBart.
    """
    inputs = tokenizer(text_block, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]
    output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def _translate_block_Helsinki(text_block, model, tokenizer, device):
    inputs = tokenizer(text_block, return_tensors="pt", padding = True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def _translate_block_nllb(text_block, model, tokenizer, device):
    """
    Traduce un bloque de texto del español al inglés usando NLLB-200.
    """
    if not text_block.strip():
        return ""

    # Codificar texto indicando el idioma de origen
    inputs = tokenizer(
        text_block,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forzar idioma de destino (inglés latino)
    #forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]

    # Generar traducción
    output = model.generate(
        **inputs,
        #forced_bos_token_id=forced_bos_token_id
    )

    # Decodificar resultado
    return tokenizer.decode(output[0], skip_special_tokens=True)


class translator():
    def __init__(self, model, tokenizer, _translate_block_function):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Cargar modelo y tokenizer
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        # Configurar idioma de origen
        self.tokenizer.src_lang = 'spa_Latn'
        self.tokenizer.tgt_lang = 'eng_Latn'
        # Guardar función de traducción
        self._translate_block = _translate_block_function

    def translate_es_to_en(self, text):
        try:
            """
            Traduce un texto largo del español al inglés usando NLLB-200-1.3B,
            dividiendo en bloques seguros de hasta 1024 tokens (post-tokenización).
            """
            if not text.strip():
                return ""

            # Dividir por saltos de línea
            paragraphs = text.split('\n')
            translated_blocks = []

            for idx, para in enumerate(paragraphs):
                if para.strip():
                    translated = self._translate_block(para, self.model, self.tokenizer, self.device)
                    translated_blocks.append(translated)

            return "\n".join(translated_blocks)
        
        except Exception as e:
            print(f"Error al traducir: {e}")
            return "problems with text"

    # Detección y traducción
    def detect_and_translate(self, text):
        lang, _ = langid.classify(text)
        if lang == 'es':
            return self.translate_es_to_en(text)
        return text



"""
# Ejemplo de uso en dataframe
import pandas as pd

df = pd.read_excel("data.xlsx")
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
_translate_block_function = lambda text: _translate_block_bart(text, model, tokenizer, model.device)

trans = translator(model, tokenizer, _translate_block_function)
# Aplicar traducción a la columna 'text_for_embedding'
df["text_for_embedding_translated"] = df.apply(
    lambda r: trans.detect_and_translate(r.get("text_for_embedding", "")),
    axis=1
)
df.to_excel("data_translated.xlsx", index=False)

"""




