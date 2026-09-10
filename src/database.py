import sqlite3
from pathlib import Path

DB_PATH = Path("prediction_history.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        predicted_class TEXT,
        confidence REAL,
        top_predictions TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit()
    con.close()

def save_prediction(filename, predicted_class, confidence, top_predictions):
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO predictions(filename,predicted_class,confidence,top_predictions) VALUES (?,?,?,?)",
                (filename, predicted_class, confidence, top_predictions))
    con.commit()
    con.close()

def recent_predictions(limit=20):
    init_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT filename,predicted_class,confidence,top_predictions,created_at FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return rows
