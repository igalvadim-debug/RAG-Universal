import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

# Путь к твоей модели bge-m3
MODEL_PATH = r"C:\Users\Startklar\Desktop\zaebalo_mirror\RAG"

# Папка с твоими output_*.md
MD_DIR = r"C:\Users\Startklar\Desktop\zaebalo_mirror"

# Папка для Chroma
DB_DIR = r"C:\Users\Startklar\Desktop\zaebalo_mirror\chroma_zaebalo"

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
