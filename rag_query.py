"""
rag_query.py — CLI-режим RAG.
Использует rag_core.py (относительные пути, llama.cpp + KoboldCpp).

Запуск:
    python rag_query.py
"""

from rag_core import (
    search_chunks, build_context,
    generate_rag_answer, generate_topics,
    get_backend, backend_status,
    scan_gguf, start_llama_server,
)
from pathlib import Path

BASE_DIR = Path(__file__).parent


def main():
    print("═" * 50)
    print("  RAG-чат · zaebalo.ru")
    print("═" * 50)

    # Проверяем бэкенд
    backend = get_backend()
    if backend == "none":
        print("\n⚠️  Ни KoboldCpp (5001), ни llama-server (8080) не запущены.")
        print("Хочешь запустить llama-server? Введи путь к папке с GGUF или Enter чтобы пропустить:")
        folder = input("  Папка: ").strip()
        if folder:
            models = scan_gguf(folder)
            if models:
                print("\nНайденные модели:")
                for i, m in enumerate(models):
                    print(f"  [{i}] {m}")
                idx = input("Выбери номер: ").strip()
                try:
                    chosen = models[int(idx)]
                    print(f"Запускаю: {chosen}")
                    ok = start_llama_server(chosen)
                    if not ok:
                        print("Не удалось запустить. Продолжаю без LLM (только поиск).")
                except (ValueError, IndexError):
                    print("Неверный номер. Продолжаю без LLM.")
            else:
                print("GGUF-файлы не найдены.")
    else:
        print(f"\n✅ Бэкенд: {backend.upper()}")

    print("\nКоманды: темы | стат | выход")
    print("─" * 50)

    while True:
        try:
            q = input("\nВопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not q:
            continue

        if q.lower() in ("выход", "exit", "quit"):
            break

        # Статистика
        if q.lower() in ("стат", "статистика"):
            from stats import get_stats
            print(get_stats())
            continue

        # 10 тем
        if q.lower() in ("темы", "10 тем", "дай темы", "предложи темы"):
            print("\n⏳ Генерирую темы...")
            result = generate_topics("жалобы на жизнь")
            print("\n═══ 10 тем ═══\n")
            print(result)
            continue

        # Обычный RAG
        print("\n⏳ Ищу релевантные чанки...")
        chunks  = search_chunks(q)
        context = build_context(chunks)
        answer  = generate_rag_answer(q, context, style="реализм", word_limit=200)

        print(f"\n📎 Найдено чанков: {len(chunks)}")
        print("\n═══ Ответ ═══\n")
        print(answer)
        print("\n" + "─" * 50)

    print("\nПока!")


if __name__ == "__main__":
    main()
