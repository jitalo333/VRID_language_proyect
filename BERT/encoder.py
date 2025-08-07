import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import pandas as pd

def embed_texts(texts, BASE_MODEL, ADAPTER_NAME, batch_size=32, device='cpu'):
    #Parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.load_adapter(ADAPTER_NAME, source="hf", set_active=True)
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

def prepare_data(df):
  texts = df["text_for_embedding_translated"].tolist()
  labels = df["Interdisciplinario"]
  #Eliminar labels = indefinido
  labels = labels[labels != "INDEFINIDO"]
  #label encoder
  le = LabelEncoder()
  labels = le.fit_transform(labels)
  labels = pd.Series(labels, index=df[df["Interdisciplinario"] != "INDEFINIDO"].index)
  texts = [texts[i] for i in labels.index]
  return texts, labels