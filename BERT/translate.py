import langid
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class translator():
  def __init__(self):
    # Cargar modelo y tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
    # Configurar idioma de origen
    self.tokenizer.src_lang = "es_XX"

    # Obtener el token BOS (beginning of sentence) para inglés
    self.forced_bos_token_id = self.tokenizer.lang_code_to_id["en_XX"]

  def translate_es_to_en_nllb(self, text, max_tokens=1024):
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
            translated = self._translate_block(para)
            translated_blocks.append(translated)

        return "\n".join(translated_blocks)
    except Exception as e:
        print(f"Error al traducir: {e}")
        return "problems with text"


  def _translate_block(self, text_block):
      inputs = self.tokenizer(text_block, return_tensors="pt", truncation=True, max_length=1024)
      inputs = {k: v.to(self.device) for k, v in inputs.items()}
      output = self.model.generate(**inputs, forced_bos_token_id=self.forced_bos_token_id)
      return self.tokenizer.decode(output[0], skip_special_tokens=True)

  def translate_es_en(self, text):
      batch = self.tokenizer([text], return_tensors="pt", padding=True)
      gen = self.model.generate(**batch)
      return self.tokenizer.decode(gen[0], skip_special_tokens=False)

  # Detección y traducción
  def detect_and_translate(self, text):
      lang, _ = langid.classify(text)
      if lang == 'es':
          return self.translate_es_to_en_nllb(text)
      return text

