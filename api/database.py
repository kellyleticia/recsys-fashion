import sqlite3
from pathlib import Path

DB_PATH = 'data/recsys.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def criar_tabelas():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS salvos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            product_id  INTEGER NOT NULL,
            salvo_em    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, product_id)
        );

        CREATE TABLE IF NOT EXISTS historico (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            query       TEXT NOT NULL,
            buscado_em  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    db.commit()
    db.close()
    print("Banco de dados pronto")