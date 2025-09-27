import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


def embed_texts(texts, BASE_MODEL, ADAPTER_NAME, batch_size=32, device='cpu'):
    #Parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.load_adapter(ADAPTER_NAME, source="hf", set_active=True, load_as="classification")
    model.to(device)
    model.eval()
    print("Modelo SPECTER2 cargado correctamente.")
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=512).to(device)
        with torch.no_grad():
            out = model(**encoded)
            embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        embeddings.append(embs.cpu().numpy())
    return np.vstack(embeddings)



class BERT_vectorizer:
    def __init__(self, BASE_MODEL, ADAPTER_NAME):
        # Inicializa con el modelo base y el adapter a usar
        self.BASE_MODEL = BASE_MODEL
        self.ADAPTER_NAME = ADAPTER_NAME

    def transform(self, X):
        # Genera embeddings para los textos usando embed_texts, 
        # compatible con Pipeline de sklearn
        return embed_texts(X, self.BASE_MODEL, self.ADAPTER_NAME)