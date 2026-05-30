import numpy as np
import faiss

print("Carregando embeddings...")
embeddings = np.load('data/embeddings.npy')
print(f"Shape: {embeddings.shape}")

print("Normalizando...")
faiss.normalize_L2(embeddings)

print("Construindo índice FAISS...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Vetores indexados: {index.ntotal}")

faiss.write_index(index, 'data/index.faiss')
print("Salvo em data/index.faiss")