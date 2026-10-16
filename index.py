from pathlib import Path
import re

index_path = Path(r"C:\Users\Startklar\Desktop\zaebalo_mirror\zaebalo.ru\index.html")

text = index_path.read_text(encoding="utf-8")

# Заменяем href="/?page=N" на href="?page=N"
text = re.sub(r'href="\/\?page=(\d+)"', r'href="?page=\1"', text)

index_path.write_text(text, encoding="utf-8")

print("Пагинация исправлена, теперь ссылки работают локально.")