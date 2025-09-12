from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def encoder_sentence_transformers(model_name, texts, batch_size=8, max_length=512):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)  # ya queda en GPU si existe

    # encode hace batching automáticamente, pero si quieres controlar batch_size lo pasas como argumento
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings  # (n_texts, hidden_dim)