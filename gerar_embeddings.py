import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time

print("Carregando produtos...")
with open('data/produtos.json', encoding='utf-8') as f:
    produtos = json.load(f)
print(f"{len(produtos)} produtos carregados")

print("Carregando modelo...")
modelo = SentenceTransformer('all-MiniLM-L6-v2')

textos = [p['texto_embedding'] for p in produtos]

print("Gerando embeddings...")
inicio = time.time()
embeddings = modelo.encode(
    textos,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Pronto em {round(time.time() - inicio, 1)}s!")
print(f"Shape: {embeddings.shape}")

np.save('data/embeddings.npy', embeddings)
print("Salvo em data/embeddings.npy")