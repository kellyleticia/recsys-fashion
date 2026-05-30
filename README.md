# 👗 Fashion RecoSys

> Sistema de recomendação de moda com busca semântica usando embeddings e FAISS  
> Projeto de portfólio 

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-5.1-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.13-red)

---

## Screenshots

![Busca](https://raw.githubusercontent.com/kellyleticia/recsys-fashion/main/docs/busca.png)
![Salvos](https://raw.githubusercontent.com/kellyleticia/recsys-fashion/main/docs/salvos.png)

## O que o projeto faz

O usuário descreve em linguagem natural o que procura. O sistema transforma essa descrição em um vetor semântico (embedding de 384 dimensões), compara com os vetores de **44.417 produtos** usando similaridade de cosseno via FAISS, e retorna os mais relevantes com fotos reais.

Os itens salvos ficam persistidos num banco SQLite associados a um `user_id` — os favoritos continuam disponíveis ao recarregar a página ou voltar depois.

---

## Arquitetura

```
query (texto livre)
        │
        ▼
  encode()  ──  sentence-transformers (all-MiniLM-L6-v2)
        │               384 dimensões
        ▼
  FAISS index  ──  IndexFlatIP (cosine similarity)
        │               44.417 vetores indexados
        ▼
  top-K produtos rankeados por score
        │
        ▼
  FastAPI  ──  JSON response com links das fotos
        │
        ▼
  Frontend HTML/JS  ──  grid com imagens reais + salvos no SQLite
```

---

## Stack

| Camada | Tecnologia | Motivo |
|--------|-----------|--------|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | 80MB, 384d, roda na CPU, ~11s para 44k itens |
| Busca vetorial | FAISS `IndexFlatIP` | busca exata, ideal para < 1M itens |
| API | FastAPI + Uvicorn | async nativo, docs automáticas, padrão ML serving |
| Banco de dados | SQLite | zero config, persistência real entre sessões |
| Frontend | HTML/CSS/JS puro | sem framework, fácil de auditar |

---

## Dataset

**Fashion Product Images Dataset** — Kaggle  
44.417 produtos de moda com metadados e imagens hospedadas online.

| Coluna | Descrição |
|--------|-----------|
| `id` | ID único do produto |
| `productDisplayName` | Nome completo com marca |
| `gender` | Men, Women, Boys, Girls, Unisex |
| `masterCategory` | Apparel, Accessories, Footwear... |
| `articleType` | Shirts, Jeans, Watches... |
| `baseColour` | Cor principal |
| `season` | Summer, Fall, Winter, Spring |
| `usage` | Casual, Sports, Formal, Ethnic |

O campo `texto_embedding` concatena todas as colunas em uma frase rica:

```
"Turtle Check Men Navy Blue Shirt Shirts Topwear Men Navy Blue Fall Casual"
```

Isso dá embeddings mais precisos do que indexar só o título.

---

## Como rodar localmente

### 1. Clone e sincronize as dependências

```bash
git clone https://github.com/kellyleticia/recsys-fashion
cd recsys-fashion
uv sync
```

### 2. Dataset
`styles.csv` e `images.csv` já estão na pasta `data/` do repositório.

### 3. Prepare os dados
```bash
python preparar_dados.py      # limpa os CSVs → data/produtos.json
python gerar_embeddings.py    # gera vetores  → data/embeddings.npy
python construir_indice.py    # indexa FAISS  → data/index.faiss
```

### 4. Rode a API
```bash
uvicorn api.main:app --reload
```

### 5. Abra o frontend
Abra `frontend/index.html` no navegador.

---

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Status e total de produtos |
| `POST` | `/recomendar` | Recomendação por query em linguagem natural |
| `GET` | `/salvos/{user_id}` | Produtos salvos do usuário |
| `POST` | `/salvos` | Salva ou remove um produto (toggle) |
| `GET` | `/historico/{user_id}` | Histórico de buscas |

### Exemplo
```bash
curl -X POST http://localhost:8000/recomendar \
  -H "Content-Type: application/json" \
  -d '{"query": "casual blue shirt for men", "top_k": 6}'
```

---

## Estrutura do projeto

```
recsys-fashion/
├── api/
│   ├── __init__.py
│   ├── main.py           ← endpoints FastAPI
│   └── database.py       ← SQLite: salvos e histórico
├── data/
│   ├── styles.csv        ← dataset original
│   ├── images.csv        ← URLs das imagens
│   ├── produtos.json     ← dados processados
│   ├── embeddings.npy    ← vetores 44417 × 384 (gerado localmente)
│   └── index.faiss       ← índice de busca vetorial (gerado localmente)
├── docs/
│   ├── busca.png         ← screenshot da busca
│   └── salvos.png        ← screenshot dos salvos
├── frontend/
│   └── index.html        ← UI completa
├── notebooks/
│   └── analise.ipynb     ← EDA e visualizações
├── .gitignore
├── .python-version
├── construir_indice.py
├── gerar_embeddings.py
├── preparar_dados.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Decisões técnicas

**Por que sentence-transformers em vez de TF-IDF?**  
TF-IDF conta palavras. Se o usuário busca "navy top" e o produto se chama "Shirts Topwear Men Blue", TF-IDF não conecta. Embeddings entendem significado — "top" e "Topwear" vivem próximos no espaço vetorial.

**Por que FAISS em vez de busca linear?**  
Busca linear em 44k vetores de 384 dimensões seria ~17M operações por query. FAISS retorna os vizinhos em < 5ms.

**Por que IndexFlatIP?**  
Faz busca exata. Para 44k itens é rápido o suficiente. Para milhões de itens, a troca seria para `IndexIVFFlat` ou `IndexHNSWFlat` com pequena perda de precisão.

**Por que SQLite?**  
Zero configuração para portfólio. Em produção seria PostgreSQL — o código da API não mudaria, só a connection string.
