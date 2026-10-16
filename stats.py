"""
stats.py — статистика ChromaDB и папки docs.
"""

import os
from pathlib import Path
import chromadb

BASE_DIR    = Path(__file__).parent
CHROMA_DIR  = BASE_DIR / "chroma_zaebalo"
DOCS_DIR    = BASE_DIR / "docs"
COLLECTION  = "zaebalo"


def get_stats() -> str:
    """Возвращает строку со статистикой базы."""
    lines = []

    # ── ChromaDB ──────────────────────────────
    try:
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(COLLECTION)
        total_chunks = collection.count()

        # Уникальные файлы в метаданных
        meta = collection.get(include=["metadatas"])["metadatas"]
        files_in_db = {m["file"] for m in meta if m and "file" in m}

        # Общий объём текста
        docs = collection.get(include=["documents"])["documents"]
        total_chars = sum(len(d) for d in docs if d)
        total_words = sum(len(d.split()) for d in docs if d)

    except Exception as e:
        return f"[Ошибка] Не могу открыть ChromaDB: {e}"

    # ── Размер базы на диске ──────────────────
    db_size = 0
    if CHROMA_DIR.exists():
        for f in CHROMA_DIR.rglob("*"):
            if f.is_file():
                db_size += f.stat().st_size

    # ── Папка docs ────────────────────────────
    docs_files = list(DOCS_DIR.glob("*.md")) if DOCS_DIR.exists() else []
    docs_size  = sum(f.stat().st_size for f in docs_files)

    lines.append("═" * 40)
    lines.append("         СТАТИСТИКА RAG-БАЗЫ")
    lines.append("═" * 40)
    lines.append(f"  Чанков в базе:       {total_chunks:>8,}")
    lines.append(f"  Файлов проиндекс.:   {len(files_in_db):>8,}")
    lines.append(f"  MD-файлов в /docs/:  {len(docs_files):>8,}")
    lines.append(f"  Слов в базе:         {total_words:>8,}")
    lines.append(f"  Символов в базе:     {total_chars:>8,}")
    lines.append(f"  Размер ChromaDB:     {db_size / 1024 / 1024:>7.1f} МБ")
    lines.append(f"  Размер /docs/:       {docs_size / 1024 / 1024:>7.1f} МБ")
    lines.append("═" * 40)

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_stats())
