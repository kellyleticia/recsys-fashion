from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from api.database import get_db, criar_tabelas

print("Carregando dados...")
with open('data/produtos.json', encoding='utf-8') as f:
    produtos = json.load(f)

produtos_por_id = {p['id']: p for p in produtos}

print("Carregando modelo e índice...")
modelo = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('data/index.faiss')
print(f"Pronto! {len(produtos)} produtos indexados")

app = FastAPI(title="RecoSys Fashion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    criar_tabelas()

class BuscaRequest(BaseModel):
    query: str
    user_id: str = "anonimo"
    top_k: int = 24
    categoria: str = "all"

class SalvarRequest(BaseModel):
    user_id: str
    product_id: int

@app.get("/")
def root():
    return {"status": "online", "produtos": len(produtos)}

@app.post("/recomendar")
def recomendar(req: BuscaRequest):
    db = get_db()
    db.execute(
        "INSERT INTO historico (user_id, query) VALUES (?, ?)",
        (req.user_id, req.query)
    )
    db.commit()
    db.close()

    vec = modelo.encode([req.query], convert_to_numpy=True)
    faiss.normalize_L2(vec)

    k = req.top_k * 5 if req.categoria != "all" else req.top_k
    scores, indices = index.search(vec, min(k, len(produtos)))

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        produto = produtos[idx].copy()
        if req.categoria != "all":
            if produto.get("masterCategory", "").lower() != req.categoria.lower():
                continue
        produto["score"] = round(float(score) * 100, 1)
        resultados.append(produto)
        if len(resultados) >= req.top_k:
            break

    return {"resultados": resultados, "total": len(resultados), "query": req.query}

@app.post("/salvos")
def salvar(req: SalvarRequest):
    """Salva um produto para o usuário."""
    db = get_db()
    try:
        db.execute(
            "INSERT INTO salvos (user_id, product_id) VALUES (?, ?)",
            (req.user_id, req.product_id)
        )
        db.commit()
        return {"status": "salvo"}
    except Exception:
        db.execute(
            "DELETE FROM salvos WHERE user_id = ? AND product_id = ?",
            (req.user_id, req.product_id)
        )
        db.commit()
        return {"status": "removido"}
    finally:
        db.close()

@app.get("/salvos/{user_id}")
def get_salvos(user_id: str):
    """Retorna todos os produtos salvos pelo usuário."""
    db = get_db()
    rows = db.execute(
        "SELECT product_id, salvo_em FROM salvos WHERE user_id = ? ORDER BY salvo_em DESC",
        (user_id,)
    ).fetchall()
    db.close()

    salvos = []
    for row in rows:
        produto = produtos_por_id.get(row["product_id"])
        if produto:
            salvos.append({**produto, "salvo_em": row["salvo_em"]})

    return {"salvos": salvos, "total": len(salvos)}

@app.get("/historico/{user_id}")
def get_historico(user_id: str):
    """Retorna histórico de buscas do usuário."""
    db = get_db()
    rows = db.execute(
        "SELECT query, buscado_em FROM historico WHERE user_id = ? ORDER BY buscado_em DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    db.close()
    return {"historico": [dict(r) for r in rows]}