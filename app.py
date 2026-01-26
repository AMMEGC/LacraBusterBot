def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    # Tabla base (mínimo)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ocr_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            message_id INTEGER,
            file_unique_id TEXT,
            created_at TEXT,
            ocr_text TEXT
        )
    """)

    # --- Migración segura: agrega columnas si faltan ---
    cur.execute("PRAGMA table_info(ocr_texts)")
    cols = {row[1] for row in cur.fetchall()}

    def add_col(name: str, coltype: str):
        if name not in cols:
            cur.execute(f"ALTER TABLE ocr_texts ADD COLUMN {name} {coltype}")

    # Las que tu código YA está intentando usar
    add_col("image_hash", "TEXT")
    add_col("ocr_status", "TEXT")      # <- esta te está faltando
    add_col("ocr_exit_code", "INTEGER")
    add_col("ocr_error", "TEXT")

    # Índices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_file_unique_id ON ocr_texts(file_unique_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_image_hash ON ocr_texts(image_hash)")

    conn.commit()
    conn.close()

