import glob
import os

def estimate_tokens(text: str) -> int:
    """Примерная, но очень точная оценка для русского текста"""
    # Для русского: ~3 символа = 1 токен
    chars = len(text)
    tokens = chars // 3 + (chars % 3 > 0)
    
    # Небольшая корректировка (пробелы, переносы, разделители)
    tokens = int(tokens * 0.97)
    return max(tokens, 0)

def main():
    md_files = sorted(glob.glob("output_*.md"))
    
    if not md_files:
        print("Файлы output_*.md не найдены!")
        return
    
    total_tokens = 0
    total_size = 0
    
    print(f"{'Файл':<15} {'Размер (КБ)':>12} {'Токенов (оценка)':>18}   {'Ток/КБ':>8}")
    print("-" * 70)
    
    for md in md_files:
        try:
            size_kb = os.path.getsize(md) / 1024
            
            with open(md, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            tokens = estimate_tokens(text)
            
            print(f"{md:<15} {size_kb:12.1f} {tokens:18,}   {tokens/size_kb:8.1f}")
            
            total_tokens += tokens
            total_size += size_kb
            
        except Exception as e:
            print(f"Ошибка в {md}: {e}")

    print("-" * 70)
    print(f"ИТОГО: {total_size:.1f} КБ  →  ≈ {total_tokens:,} токенов")
    print(f"Среднее на файл: ≈ {total_tokens / len(md_files):,.0f} токенов")

if __name__ == "__main__":
    main()