import os
from bs4 import BeautifulSoup
import glob

def extract_post_text(html_content):
    """Извлекает только текст постов из div.item"""
    soup = BeautifulSoup(html_content, 'html.parser')

    posts = []
    for item in soup.find_all('div', class_='item'):
        content_div = item.find('div', align='left')
        if content_div:
            text = content_div.get_text(separator='\n', strip=True)

            # Убираем лишние пустые строки
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n\n'.join(lines)

            if clean_text:
                posts.append(clean_text)

    return posts


def main():
    html_dir = r'C:\Users\Startklar\Desktop\zaebalo_mirror\zaebalo.ru'

    # Ищем все html-файлы рекурсивно
    html_files = glob.glob(os.path.join(html_dir, '**', '*.html'), recursive=True)
    html_files.extend(glob.glob(os.path.join(html_dir, '**', '*.htm'), recursive=True))

    if not html_files:
        print("HTML-файлы не найдены!")
        return

    print(f"Найдено HTML-файлов: {len(html_files)}")

    separator = '\n' + '-' * 75 + '\n\n'
    max_size = 20 * 1024  # 15 КБ
    file_counter = 1
    current_md_path = f'output_{file_counter:03d}.md'
    current_content = []
    current_size = 0

    for html_file in sorted(html_files):
        try:
            with open(html_file, 'r', encoding='windows-1251', errors='replace') as f:
                html_content = f.read()

            posts = extract_post_text(html_content)

            for post_text in posts:

                # Фильтр: пропускаем посты, где меньше 35 слов
                if len(post_text.split()) < 45:
                    continue

                block = post_text + separator
                block_size = len(block.encode('utf-8'))

                # Если блок не помещается — создаём новый файл
                if current_content and (current_size + block_size > max_size):
                    with open(current_md_path, 'w', encoding='utf-8') as f:
                        f.write(''.join(current_content))
                    print(f"Создан {current_md_path} ({current_size / 1024:.1f} КБ)")

                    file_counter += 1
                    current_md_path = f'output_{file_counter:03d}.md'
                    current_content = []
                    current_size = 0

                current_content.append(block)
                current_size += block_size

        except Exception as e:
            print(f"Ошибка при обработке {html_file}: {e}")

    # Сохраняем последний файл
    if current_content:
        with open(current_md_path, 'w', encoding='utf-8') as f:
            f.write(''.join(current_content))
        print(f"Создан {current_md_path} ({current_size / 1024:.1f} КБ)")


if __name__ == "__main__":
    main()
