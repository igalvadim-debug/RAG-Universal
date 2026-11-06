import os
import glob
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent

# Путь к модели эмбеддингов (paraphrase-multilingual-MiniLM-L12-v2 из models.txt:
# D:\ComfyUi\models\LLM\paraphrase-multilingual-MiniLM-L12-v2).
# Локальная копия model/ имеет приоритет, иначе — папка из ComfyUI.
COMFY_EMBED_MODEL = Path(r"D:\ComfyUi\models\LLM\paraphrase-multilingual-MiniLM-L12-v2")
MODEL_PATH = str(BASE_DIR / "model") if (BASE_DIR / "model").exists() else str(COMFY_EMBED_MODEL)

# Папка с твоими output_*.md (зеркало zaebalo.ru)
MD_DIR = str(BASE_DIR)

# Папка для Chroma (совпадает с RAG_DB по умолчанию в rag_config.py)
DB_DIR = os.environ.get("RAG_DB", str(BASE_DIR / "chroma_zaebalo"))

CHUNK_SIZE = 1500  # символов


def chunk_text(text, size=CHUNK_SIZE):
    for i in range(0, len(text), size):
        yield text[i:i+size]


def main():
    print("Загружаю модель эмбеддингов...")
    model = SentenceTransformer(MODEL_PATH)

    print("Создаю Chroma-базу...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("zaebalo")

    md_files = sorted(glob.glob(os.path.join(MD_DIR, "output_*.md")))
    print(f"Найдено файлов: {len(md_files)}")

    doc_id = 0

    for md in md_files:
        print(f"Обрабатываю: {md}")
        with open(md, "r", encoding="utf-8") as f:
            text = f.read()

        for chunk in chunk_text(text):
            if len(chunk.strip()) < 50:
                continue

            emb = model.encode(chunk).tolist()

            collection.add(
                ids=[str(doc_id)],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"file": md}]
            )

            doc_id += 1

    print("\nГотово! Индексация завершена.")


if __name__ == "__main__":
    main()
