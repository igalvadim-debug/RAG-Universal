"""
update_db.py — индексирует MD-файлы из ./docs/ в ChromaDB.
Чанкинг по разделителю '---...---' (как делает copilot.py).
Уже проиндексированные файлы не трогает.
"""

import os
import glob
import hashlib
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# ПУТИ
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / "model"
CHROMA_DIR  = Path(os.environ.get("RAG_DB", str(BASE_DIR / "chroma_zaebalo")))
DOCS_DIR    = BASE_DIR / "docs"

COLLECTION  = os.environ.get("RAG_COLLECTION", "zaebalo")
MIN_WORDS   = 20       # чанки короче — пропускаем
SEP         = "-" * 75 # разделитель из copilot.py


def chunk_by_separator(text: str) -> list[str]:
    """
    Режет текст по разделителю '---...---' (75 дефисов).
    Возвращает непустые чанки.
    """
    parts = text.split(SEP)
    result = []
    for part in parts:
        clean = part.strip()
        if clean and len(clean.split()) >= MIN_WORDS:
            result.append(clean)
    return result


def file_hash(path: str) -> str:
    """MD5-хэш файла — используется как уникальный ID."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_indexed_sources(collection) -> set[str]:
    """Возвращает множество путей файлов уже в базе."""
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        return {m["file"] for m in all_meta if m and "file" in m}
    except Exception:
        return set()


def main(log_callback=None):
    """
    Сканирует DOCS_DIR, индексирует новые MD-файлы.
    log_callback — функция для передачи лога в Gradio (опционально).
    """
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    if not DOCS_DIR.exists():
        log(f"[DB] Папка {DOCS_DIR} не найдена. Создаю...")
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        log("[DB] Положи MD-файлы в ./docs/ и запусти снова.")
        return

    log(f"[DB] Загружаю модель из {MODEL_DIR} ...")
    model = SentenceTransformer(str(MODEL_DIR))

    log("[DB] Подключаюсь к ChromaDB ...")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION)

    # Уже проиндексированные файлы
    indexed = get_indexed_sources(collection)
    log(f"[DB] Уже в базе файлов: {len(indexed)}")

    # Ищем MD-файлы
    md_files = sorted(glob.glob(str(DOCS_DIR / "*.md")))
    log(f"[DB] Найдено MD-файлов в ./docs/: {len(md_files)}")

    new_count   = 0
    chunk_count = 0

    for md_path in md_files:
        # Пропускаем уже проиндексированные
        if md_path in indexed:
            log(f"[DB] Пропускаю (уже есть): {os.path.basename(md_path)}")
            continue

        log(f"[DB] Индексирую: {os.path.basename(md_path)}")

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_by_separator(text)
        log(f"[DB]   → {len(chunks)} чанков")

        fhash = file_hash(md_path)

        for i, chunk in enumerate(chunks):
            doc_id = f"{fhash}_{i}"
            emb    = model.encode(chunk).tolist()

            collection.add(
                ids        = [doc_id],
                embeddings = [emb],
                documents  = [chunk],
                metadatas  = [{"file": md_path, "chunk_idx": i}],
            )
            chunk_count += 1

        new_count += 1

    log(f"\n[DB] Готово! Добавлено файлов: {new_count}, чанков: {chunk_count}")
    log(f"[DB] Всего в базе: {collection.count()} чанков")


if __name__ == "__main__":
    main()
