"""
webui.py — Gradio-интерфейс для RAG-проекта.

Запуск:
    python webui.py
    или через bat:
    start.bat
"""

import gradio as gr
from pathlib import Path
import threading
import shutil
import random
import string
import subprocess
import json

import rag_core
import update_db
import stats
import project_manager

BASE_DIR     = Path(__file__).parent
UPLOAD_CACHE = BASE_DIR / "_upload_cache"
UPLOAD_CACHE.mkdir(exist_ok=True)


def get_music_style_choices() -> list[str]:
    """Liest stylemusic.json und gibt alle 'music: ...' Keys als Liste zurück."""
    style_path = BASE_DIR / "stylemusic.json"
    if not style_path.exists():
        return []
    try:
        data = json.loads(style_path.read_text(encoding="utf-8"))
        styles = data.get("ltx_music_styles", [])
    except Exception:
        return []
    result = []
    for cat in styles:
        for name in cat.get("styles", []):
            result.append(f"music: {name}")
    return result


def _safe_copy_image(path: str | None, slot_idx: int) -> str | None:
    """
    Gradio 5.x сохраняет загруженные файлы во временную папку uvicorn
    и иногда удаляет их до того как наш код их читает.
    Копируем файл в стабильный _upload_cache/ при получении.
    Возвращает путь к стабильной копии или None.
    """
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        print(f"[RAG] ⚠️ Слот {slot_idx}: файл не найден: {path}")
        return None
    try:
        dst = UPLOAD_CACHE / f"slot_{slot_idx}{src.suffix}"
        shutil.copy2(src, dst)
        return str(dst)
    except Exception as e:
        print(f"[RAG] ⚠️ Слот {slot_idx}: ошибка копирования: {e}")
        return str(src)  # фолбэк — оригинальный путь

# ──────────────────────────────────────────────
# СОСТОЯНИЕ
# ──────────────────────────────────────────────
_gguf_list   = []   # список найденных GGUF
_last_chunks = []   # последние найденные чанки


# ──────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ
# ──────────────────────────────────────────────

def backend_status() -> str:
    b = rag_core.get_backend()
    if b == "kobold":
        return "✅ KoboldCpp (порт 5001)"
    if b == "llama":
        return "✅ llama-server (порт 8080)"

    env = rag_core.load_env()
    clouds = []
    for prov_name, cfg in rag_core.CLOUD_PROVIDERS.items():
        if env.get(cfg["key_var"], "").strip():
            clouds.append(f"{prov_name} ({cfg['model']})")

    if clouds:
        return f"☁️ Локальный сервер не запущен. Доступны облачные API: {', '.join(clouds)}"
    return "⛔ Нет ни локального сервера, ни облачных API-ключей в .env"


def scan_models(folder: str):
    global _gguf_list
    _gguf_list = rag_core.scan_gguf(folder)
    if not _gguf_list:
        env = rag_core.load_env()
        has_cloud = any(env.get(cfg["key_var"], "").strip() for cfg in rag_core.CLOUD_PROVIDERS.values())
        msg = f"⚠️ GGUF-модели не найдены в: {folder}"
        if has_cloud:
            msg += " (облачные API доступны — генерация работает без локальной модели)"
        return gr.update(choices=[], value=None), msg
    return gr.update(choices=_gguf_list, value=_gguf_list[0]), f"✅ Найдено моделей: {len(_gguf_list)}"


def launch_llama(model_path: str):
    if not model_path:
        env = rag_core.load_env()
        has_cloud = any(env.get(cfg["key_var"], "").strip() for cfg in rag_core.CLOUD_PROVIDERS.values())
        if has_cloud:
            return "ℹ️ Нет локальной модели, но облачные API доступны."
        return "⚠️ Сначала выбери модель из списка."

    log_lines = []
    def cb(msg):
        log_lines.append(msg)

    ok = rag_core.start_llama_server(model_path, log_callback=cb)
    result = "\n".join(log_lines)
    if ok:
        return result + "\n✅ llama-server запущен!"
    return result + "\n⛔ Не удалось запустить llama-server."


def load_system_prompt(path: str) -> tuple[str, str]:
    path = path.strip()
    if not path:
        return "", "⚠️ Путь не указан."
    p = Path(path)
    if not p.exists():
        return "", f"⚠️ Файл не найден: {path}"
    try:
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            import json
            data = json.loads(text)
            text = data.get("system") or data.get("prompt") or json.dumps(data, ensure_ascii=False)
        return text, f"✅ Загружено {len(text)} символов из: {p.name}"
    except Exception as e:
        return "", f"⚠️ Ошибка чтения: {e}"


def do_generate(question: str, style: str, word_limit: int, mode: str,
                language: str, script_mode: str, narrator: str,
                director_mode: str, n_sequences: int,
                cloud_provider: str, sys_prompt: str,
                img0, img1, img2, img3, img4, img5, img6, img7, img8, img9,
                cb_0_0, cb_0_1, cb_0_2, cb_0_3, cb_0_4, cb_0_5, cb_0_6, cb_0_7, cb_0_8, cb_0_9,
                cb_1_0, cb_1_1, cb_1_2, cb_1_3, cb_1_4, cb_1_5, cb_1_6, cb_1_7, cb_1_8, cb_1_9,
                cb_2_0, cb_2_1, cb_2_2, cb_2_3, cb_2_4, cb_2_5, cb_2_6, cb_2_7, cb_2_8, cb_2_9,
                cb_3_0, cb_3_1, cb_3_2, cb_3_3, cb_3_4, cb_3_5, cb_3_6, cb_3_7, cb_3_8, cb_3_9,
                cb_4_0, cb_4_1, cb_4_2, cb_4_3, cb_4_4, cb_4_5, cb_4_6, cb_4_7, cb_4_8, cb_4_9,
                cb_5_0, cb_5_1, cb_5_2, cb_5_3, cb_5_4, cb_5_5, cb_5_6, cb_5_7, cb_5_8, cb_5_9,
                cb_6_0, cb_6_1, cb_6_2, cb_6_3, cb_6_4, cb_6_5, cb_6_6, cb_6_7, cb_6_8, cb_6_9,
                cb_7_0, cb_7_1, cb_7_2, cb_7_3, cb_7_4, cb_7_5, cb_7_6, cb_7_7, cb_7_8, cb_7_9,
                cb_8_0, cb_8_1, cb_8_2, cb_8_3, cb_8_4, cb_8_5, cb_8_6, cb_8_7, cb_8_8, cb_8_9,
                cb_9_0, cb_9_1, cb_9_2, cb_9_3, cb_9_4, cb_9_5, cb_9_6, cb_9_7, cb_9_8, cb_9_9):
    if not question.strip():
        return "⚠️ Введи вопрос или тему.", "", "", backend_status(), gr.update(visible=False)

    if script_mode == "сценарий":
        eff_script_mode = "сценарий"
    elif mode == "аналитический":
        eff_script_mode = "аналитический"
    else:
        eff_script_mode = "художественный"

    technique_names = ["Dutch Angle", "Dolly Zoom", "Push-In", "Low-Angle", "POV", "Rack Focus", "Static Long Take", "Whip Pan", "Snap Zoom", "Overhead Shot"]
    cb_grid = [
        [cb_0_0, cb_0_1, cb_0_2, cb_0_3, cb_0_4, cb_0_5, cb_0_6, cb_0_7, cb_0_8, cb_0_9],
        [cb_1_0, cb_1_1, cb_1_2, cb_1_3, cb_1_4, cb_1_5, cb_1_6, cb_1_7, cb_1_8, cb_1_9],
        [cb_2_0, cb_2_1, cb_2_2, cb_2_3, cb_2_4, cb_2_5, cb_2_6, cb_2_7, cb_2_8, cb_2_9],
        [cb_3_0, cb_3_1, cb_3_2, cb_3_3, cb_3_4, cb_3_5, cb_3_6, cb_3_7, cb_3_8, cb_3_9],
        [cb_4_0, cb_4_1, cb_4_2, cb_4_3, cb_4_4, cb_4_5, cb_4_6, cb_4_7, cb_4_8, cb_4_9],
        [cb_5_0, cb_5_1, cb_5_2, cb_5_3, cb_5_4, cb_5_5, cb_5_6, cb_5_7, cb_5_8, cb_5_9],
        [cb_6_0, cb_6_1, cb_6_2, cb_6_3, cb_6_4, cb_6_5, cb_6_6, cb_6_7, cb_6_8, cb_6_9],
        [cb_7_0, cb_7_1, cb_7_2, cb_7_3, cb_7_4, cb_7_5, cb_7_6, cb_7_7, cb_7_8, cb_7_9],
        [cb_8_0, cb_8_1, cb_8_2, cb_8_3, cb_8_4, cb_8_5, cb_8_6, cb_8_7, cb_8_8, cb_8_9],
        [cb_9_0, cb_9_1, cb_9_2, cb_9_3, cb_9_4, cb_9_5, cb_9_6, cb_9_7, cb_9_8, cb_9_9],
    ]
    timeline_techniques = []
    for c in range(10):
        selected = None
        for r in range(10):
            if cb_grid[r][c]:
                selected = technique_names[r]
                break
        timeline_techniques.append(selected)

    chunks  = rag_core.search_chunks(question)
    context = rag_core.build_context(chunks)
    answer  = rag_core.generate_rag_answer(
        question            = question,
        context             = context,
        style               = style,
        language            = language,
        script_mode         = eff_script_mode,
        narrator            = narrator,
        word_limit          = int(word_limit),
        sys_prompt          = (sys_prompt or "").strip(),
        timeline_techniques = timeline_techniques,
        cloud_provider      = cloud_provider,
        director_mode       = director_mode,
        n_sequences         = int(n_sequences) if n_sequences else 5,
        image_paths         = [img0, img1, img2, img3, img4, img5, img6, img7, img8, img9],
    )

    # Перевод на русский если режим сценарий
    translation = ""
    if eff_script_mode == "сценарий" and answer.strip():
        translation = rag_core.translate_to_russian(answer, cloud_provider=cloud_provider)

    global _last_chunks
    _last_chunks = chunks
    chunks_text  = "\n\n" + ("─" * 60) + "\n\n".join(chunks)
    return answer, translation, chunks_text, backend_status(), gr.update(visible=bool(translation.strip()))


def do_topics(seed: str, cloud_provider: str = "auto"):
    result = rag_core.generate_topics(seed=seed, cloud_provider=cloud_provider)
    return result, "", backend_status()


def do_update_db():
    lines = []
    def cb(msg):
        lines.append(msg)
    update_db.main(log_callback=cb)
    return "\n".join(lines)


def do_stats():
    return stats.get_stats()


def do_create_project(question: str, narrator: str, n_sequences: int, cloud_provider: str):
    """Создаёт папку проекта и генерирует FLUX-промпты."""
    n = int(n_sequences)
    path, display = project_manager.create_project(
        question       = question,
        narrator       = narrator,
        n_sequences    = n,
        cloud_provider = cloud_provider,
    )
    return path, display


def _make_export_subfolder(prompt_text: str) -> Path:
    """Erstellt Unterordner in workflow/ basierend auf erstem Satz des Prompts + Timestamp."""
    import re as _re
    from datetime import datetime as _dt

    base = Path(r"D:\claudeAgent\RAG\workflow")
    base.mkdir(parents=True, exist_ok=True)

    # Erste 4 Woerter als Slug
    words = prompt_text.strip().split()
    slug_words = []
    for w in words[:4]:
        clean = _re.sub(r'[^a-zA-Z0-9\u0400-\u04FF]', '', w)
        if clean:
            slug_words.append(clean.lower())
    slug = "_".join(slug_words) if slug_words else "scene"
    slug = slug[:40]

    ts = _dt.now().strftime("%H%M")
    folder_name = f"{slug}_{ts}"
    folder = base / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def do_export(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, prompt_text):
    """
    Exportiert Bilder + Prompt:
    • ComfyUiVid\\input\\ — Bilder (a_XXXX_.png, ...)
    • workflow\\<prompt_slug>_HHMM\\ — Bilder + video.txt/novideo.txt
    """
    comfy_dir = Path(r"D:\ComfyUiVid\input")
    comfy_dir.mkdir(parents=True, exist_ok=True)

    content = (prompt_text or "").strip()
    images  = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    slot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    letters = string.ascii_lowercase

    # Projektordner
    proj_dir = _make_export_subfolder(content) if content else Path(r"D:\claudeAgent\RAG\workflow")

    exported = 0
    for img_path, label in zip(images, slot_labels):
        if img_path:
            src = Path(img_path)
            if src.exists():
                suffix = ''.join(random.choices(letters, k=4))
                new_name = f"{label}_{suffix}_.png"
                shutil.copy2(src, comfy_dir / new_name)
                shutil.copy2(src, proj_dir / new_name)
                exported += 1

    txt_name = "video.txt" if exported > 0 else "novideo.txt"
    if content:
        (proj_dir / txt_name).write_text(content, encoding="utf-8")
        (comfy_dir / txt_name).write_text(content, encoding="utf-8")

    if exported > 0:
        return f"✅ {exported} Bilder + {txt_name} → {proj_dir.name} & ComfyUiVid\\input"
    else:
        return f"⚠️ Keine Bilder — {txt_name} → {proj_dir.name}"


def do_json_workflow():
    """Führt D:\\claudeAgent\\RAG\\workflow\\update_workflow.py aus."""
    script = Path(r"D:\claudeAgent\RAG\workflow\update_workflow.py")
    if not script.exists():
        return f"⚠️ Script nicht gefunden: {script}"
    try:
        result = subprocess.run(
            ["python", str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(script.parent),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        if not parts:
            parts.append("✅ Script beendet (keine Ausgabe).")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return "⚠️ Timeout (120s) — Script läuft zu lange."
    except Exception as e:
        return f"⚠️ Fehler: {e}"


def do_controlnet(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, prov):
    """
    Generiert ControlNet Captions (Pose + Atmosphere) aus bis zu 10 Bildern.
    """
    images = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    if not any(images):
        return "⚠️ Bitte mindestens 1 Bild laden."
    try:
        return rag_core.generate_controlnet_captions(
            image_paths    = images,
            cloud_provider = prov,
        )
    except Exception as e:
        return f"⚠️ Fehler: {e}"


def save_controlnet(captions_text, negative_prompt):
    """
    Speichert ControlNet Captions + zusaetzlichen User-Negative-Prompt
    als controlnet.txt in D:\\claudeAgent\\RAG\\workflow\\.
    """
    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    workflow_dir.mkdir(parents=True, exist_ok=True)

    if not captions_text.strip():
        return "⚠️ Keine Captions zum Speichern — erst generieren."

    lines = [captions_text.strip()]
    if negative_prompt.strip():
        lines.append("")
        lines.append("USER NEGATIVE PROMPT (zusaetzlich):")
        lines.append(negative_prompt.strip())

    out_path = workflow_dir / "controlnet.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return f"✅ Gespeichert: {out_path} ({out_path.stat().st_size} bytes)"


def export_answer_prompt(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, prompt_text, with_photos: bool):
    """
    Exportiert den GENERIERTEN Prompt + optional Bilder.
    with_photos=True  → ComfyUiVid\\input + workflow\\<slug>_HHMM\\ (Bilder + video.txt)
    with_photos=False → ComfyUiVid\\input + workflow\\<slug>_HHMM\\ (nur novideo.txt)
    """
    comfy_dir = Path(r"D:\ComfyUiVid\input")
    comfy_dir.mkdir(parents=True, exist_ok=True)

    images  = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    slot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    letters = string.ascii_lowercase
    content = (prompt_text or "").strip()

    if not content:
        return "⚠️ Kein Prompt zum Exportieren — erst generieren."

    proj_dir = _make_export_subfolder(content)

    if with_photos:
        exported = 0
        for img_path, label in zip(images, slot_labels):
            if img_path:
                src = Path(img_path)
                if src.exists():
                    suffix = ''.join(random.choices(letters, k=4))
                    new_name = f"{label}_{suffix}_.png"
                    shutil.copy2(src, comfy_dir / new_name)
                    shutil.copy2(src, proj_dir / new_name)
                    exported += 1

        txt_name = "video.txt" if exported > 0 else "novideo.txt"
        (proj_dir / txt_name).write_text(content, encoding="utf-8")
        (comfy_dir / txt_name).write_text(content, encoding="utf-8")

        if exported > 0:
            return f"✅ Промт с фото: {exported} Bilder + {txt_name} → {proj_dir.name} & ComfyUiVid\\input"
        else:
            return f"⚠️ Keine Bilder — {txt_name} → {proj_dir.name} (nur Prompt)"
    else:
        txt_name = "novideo.txt"
        (proj_dir / txt_name).write_text(content, encoding="utf-8")
        (comfy_dir / txt_name).write_text(content, encoding="utf-8")
        return f"✅ Промт ohne Fotos: {txt_name} → {proj_dir.name} & ComfyUiVid\\input ({len(content)} Zeichen)"


def update_image_slots(n_sequences: int):
    """
    Возвращает обновления видимости для image-слотов, timeline-колонок
    и чекбоксов в зависимости от выбранного количества сцен.
    """
    n = int(n_sequences)
    if n == 1:
        pcts = ["0%"]
    else:
        step = 100 // (n - 1)
        pcts = [f"{i * step}%" for i in range(n)]
        pcts[-1] = "100%"

    slot_labels = [
        f"🎥 {pcts[i]} {'(Старт)' if i == 0 else '(Финал)' if i == n-1 else ''}" if i < n else f"🎥 слот {i+1}"
        for i in range(10)
    ]

    updates = []
    for i in range(10):
        updates.append(gr.update(visible=(i < n), label=slot_labels[i]))

    col_labels = []
    for i in range(10):
        if i < n:
            pct_label = pcts[i] if i < len(pcts) else f"Ч{i+1}"
            col_labels.append(gr.update(visible=True, value=f"**{pct_label}**"))
        else:
            col_labels.append(gr.update(visible=False))

    cb_updates = []
    for r in range(10):
        for c in range(10):
            cb_updates.append(gr.update(visible=(c < n)))

    return updates + col_labels + cb_updates


# ──────────────────────────────────────────────
# ИНТЕРФЕЙС
# ──────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="RAG — Генератор сценариев") as demo:

        gr.Markdown("# 🎬 RAG — Генератор сценариев")

        # ── БЛОК: Бэкенд ──────────────────────────────────────────────
        with gr.Accordion("⚙️ Настройка сервера LLM", open=True):
            status_box = gr.Textbox(
                label       = "Статус бэкенда",
                value       = backend_status(),
                interactive = False,
            )
            with gr.Row():
                model_folder = gr.Textbox(
                    label       = "Папка с GGUF-моделями",
                    value       = str(BASE_DIR),
                    placeholder = "Например: D:\\models",
                    scale       = 4,
                )
                scan_btn = gr.Button("🔍 Сканировать", scale=1)

            scan_status = gr.Textbox(label="", interactive=False, lines=1)

            model_dd = gr.Dropdown(
                label       = "Выбери модель",
                choices     = [],
                value       = None,
                interactive = True,
            )

            with gr.Row():
                launch_btn = gr.Button("🚀 Запустить llama-server", variant="primary")
                stop_btn   = gr.Button("⏹ Остановить", variant="stop")

            launch_log = gr.Textbox(label="Лог запуска", lines=5, interactive=False)

        # ── БЛОК: Системный промпт ─────────────────────────────────────
        gr.Markdown("---")
        with gr.Accordion("📝 Системный промпт", open=False):
            with gr.Row():
                prompt_path = gr.Textbox(
                    label       = "Путь к файлу (.md / .txt / .json)",
                    placeholder = r"D:\claudeAgent\RAG\prompt.md",
                    scale       = 3,
                )
                browse_btn = gr.UploadButton(
                    label      = "📁 Обзор",
                    file_types = [".md", ".txt", ".json"],
                    scale      = 1,
                )
                load_prompt_btn = gr.Button("📂 Загрузить", scale=1)
            prompt_load_status = gr.Textbox(label="", interactive=False, lines=1)
            sys_prompt_box = gr.Textbox(
                label       = "Системный промпт (можно редактировать вручную)",
                placeholder = "Например: Ты писатель в жанре магического реализма...",
                lines       = 6,
                interactive = True,
            )

        # ── БЛОК: Генерация ────────────────────────────────────────────
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label       = "Вопрос / тема / запрос",
                    placeholder = "Напиши рассказ про усталость от работы на 200 слов",
                    lines       = 3,
                )
            with gr.Column(scale=1):
                style_dd = gr.Dropdown(
                    label   = "Стиль",
                    choices = ["реализм", "мрачный", "юмор", "поток сознания"],
                    value   = "реализм",
                )
                mode_dd = gr.Dropdown(
                    label   = "Режим",
                    choices = ["художественный", "аналитический"],
                    value   = "художественный",
                )
                lang_dd = gr.Dropdown(
                    label   = "Язык вывода",
                    choices = ["русский", "немецкий", "английский", "испанский", "польский", "нидерландский", "французский", "китайский"],
                    value   = "русский",
                )
                env = rag_core.load_env()
                cloud_choices = ["auto"]
                for pn, cfg in rag_core.CLOUD_PROVIDERS.items():
                    if env.get(cfg["key_var"], "").strip():
                        cloud_choices.append(pn)
                cloud_dd = gr.Dropdown(
                    label   = "Облачный API",
                    choices = cloud_choices,
                    value   = "auto",
                )
                script_dd = gr.Dropdown(
                    label   = "Формат LTX 2.3",
                    choices = ["нет", "сценарий"],
                    value   = "нет",
                )
                base_narrators = [
                    "короткий фильм", "рассказ девушки", "рассказ парня",
                    "рассказ старушки", "рассказ старика", "документация",
                    "хоррор", "комедия", "скетч", "sci-fi", "боевик",
                    "немое кино", "немое кино (фортепиано)", "детектив",
                    "Talkshow",
                ]
                music_narrators = get_music_style_choices()
                narrator_dd = gr.Dropdown(
                    label   = "Рассказчик / Musik:Clip",
                    choices = base_narrators + music_narrators,
                    value   = "короткий фильм",
                    visible = False,
                )
                director_dd = gr.Dropdown(
                    label   = "Director Mode",
                    choices = ["нет", "Director Mode"],
                    value   = "нет",
                    visible = False,
                )
                n_seq_dd = gr.Dropdown(
                    label   = "🔢 Количество сцен (Sequences)",
                    choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    value   = 5,
                )
                word_sl = gr.Slider(
                    label   = "Длина (слов)",
                    minimum = 50,
                    maximum = 1200,
                    step    = 25,
                    value   = 200,
                )

        # ── БЛОК: Timeline матрица приёмов ────────────────────────────
        with gr.Column(visible=False) as timeline_column:
            with gr.Accordion("🎥 Кинематографические приёмы (LTX Timeline)", open=False):
                gr.Markdown("Выбери до одного приёма на каждый отрезок фильма. Пустые части = стандартная съёмка.")

                technique_labels = [
                    "🇳🇱 Dutch Angle (голландский угол — тревога)",
                    "🌀 Dolly Zoom (Вертиго — шок, искажение)",
                    "➡️ Push-In (наезд — усиление эмоций)",
                    "⬆️ Low-Angle (снизу вверх — величие/угроза)",
                    "👁️ POV (субъективная камера — от первого лица)",
                    "🔍 Rack Focus (смена фокуса — драматическое раскрытие)",
                    "🎬 Static Long Take (статичная сцена — напряжение/ожидание)",
                    "⚡ Whip Pan (резкий разворот — комедийный тайминг)",
                    "📸 Snap Zoom (рывок зума — реакция/акцент)",
                    "🔝 Overhead Shot (вид сверху — комедийная композиция)",
                ]

                with gr.Row():
                    gr.Markdown("&nbsp;")
                    tl_col_1  = gr.Markdown("**0%**")
                    tl_col_2  = gr.Markdown("**11%**")
                    tl_col_3  = gr.Markdown("**22%**")
                    tl_col_4  = gr.Markdown("**33%**")
                    tl_col_5  = gr.Markdown("**44%**")
                    tl_col_6  = gr.Markdown("**56%**")
                    tl_col_7  = gr.Markdown("**67%**")
                    tl_col_8  = gr.Markdown("**78%**")
                    tl_col_9  = gr.Markdown("**89%**")
                    tl_col_10 = gr.Markdown("**100%**")

                cbs = []
                for r in range(10):
                    with gr.Row():
                        gr.Markdown(f"**{technique_labels[r]}**")
                        row_cbs = []
                        for c in range(10):
                            cb = gr.Checkbox(label="Ctrl", value=False, interactive=True, container=False, min_width=40)
                            row_cbs.append(cb)
                        cbs.append(row_cbs)

        # Каскадная логика: только 1 чекбокс на колонку
        for r in range(10):
            for c in range(10):
                other_outputs = [cbs[rr][c] for rr in range(10) if rr != r]
                def _make_cascade(row=r, col=c):
                    def _handler(checked):
                        updates = []
                        for rr in range(5):
                            if rr != row:
                                if checked:
                                    updates.append(gr.update(interactive=False, value=False))
                                else:
                                    updates.append(gr.update(interactive=True))
                        return updates
                    return _handler
                cbs[r][c].change(fn=_make_cascade(), inputs=[cbs[r][c]], outputs=other_outputs)

        # ── БЛОК: Мульти-кадровый таймлайн (Florence-2 + LLM) ─────────
        gr.Markdown("---")
        with gr.Accordion("🖼️ Мульти-кадровый таймлайн (T2I / I2V для LTX 2.3)", open=False):
            # ── Mode-Umschalter: Video / ControlNet ────────────────────
            with gr.Row():
                img_mode_dd = gr.Dropdown(
                    label   = "🎯 Modus",
                    choices = ["Video", "Controlnet"],
                    value   = "Video",
                    scale   = 1,
                )
                gr.Markdown(
                    "<small>**Video** = LTX 2.3 Timeline-Prompts<br>"
                    "**Controlnet** = Pose + Atmosphere fuer ComfyUI</small>",
                    scale=3,
                )

            # ── VIDEO-MODUS (bestehende Timeline-Elemente) ─────────────
            with gr.Column(visible=True) as video_col:
                gr.Markdown(
                    "Загрузи от 1 до 10 ключевых кадров. Пропущенные будут заполнены автоматически: "
                    "Florence-2 опишет кадры, LLM перепишет в LTX-промпты и сгенерирует переходы."
                )

                # ── 5 динамических image-слотов ────────────────────────
                with gr.Row():
                    img_slots = [
                        gr.Image(
                            label   = f"🎥 {['0% (Старт)', '11%', '22%', '33%', '44%', '56%', '67%', '78%', '89%', '100% (Финал)'][i]}",
                            type    = "filepath",
                            height  = 180,
                            visible = True,
                        )
                        for i in range(10)
                    ]

                with gr.Row():
                    timeline_gen_btn  = gr.Button("⚡ Сгенерировать сквозной видео-промпт", variant="primary", scale=2)
                    timeline_copy_btn = gr.Button("📋 Copy", scale=1, min_width=80)

                timeline_out = gr.Textbox(
                    label       = "🎬 Матрица промптов LTX 2.3",
                    lines       = 12,
                    interactive = False,
                    placeholder = "Здесь появятся готовые [0%]…[100%] промпты...",
                )



                # ── Перевод LTX-промпта на русский ────────────────────
                gr.Markdown("---")
                with gr.Row():
                    translate_timeline_btn = gr.Button("🌐 Перевести промпт на русский", variant="secondary", scale=1, min_width=200)
                    timeline_copy_ru_btn   = gr.Button("📋 Copy перевод", scale=1, min_width=120)
                timeline_ru_out = gr.Textbox(
                    label       = "🇷🇺 Перевод видео-промпта на русский",
                    lines       = 10,
                    interactive = False,
                    placeholder = "Нажми кнопку выше — перевод появится здесь...",
                )

            # ── CONTROLNET-MODUS ───────────────────────────────────────
            with gr.Column(visible=False) as controlnet_col:
                gr.Markdown(
                    "**ControlNet Captioning**: Florence-2 analysiert jedes Bild, "
                    "LLM extrahiert POSE + ATMOSPHERE als komma-separierte Tags."
                )

                # ── Image-Slots (eigene, unabhaengig vom Video-Modus) ──
                with gr.Row():
                    cn_img_slots = [
                        gr.Image(
                            label   = f"🎥 {['0% (Start)', '11%', '22%', '33%', '44%', '56%', '67%', '78%', '89%', '100% (Final)'][i]}",
                            type    = "filepath",
                            height  = 180,
                            visible = True,
                        )
                        for i in range(10)
                    ]

                with gr.Row():
                    cn_gen_btn = gr.Button("🎭 ControlNet Captions generieren", variant="primary", scale=2)
                    cn_copy_btn = gr.Button("📋 Copy", scale=1, min_width=80)

                cn_out = gr.Textbox(
                    label       = "🎭 ControlNet Captions (Pose + Atmosphere)",
                    lines       = 14,
                    interactive = False,
                    placeholder = "POSE: standing, arms crossed, ...\nATMOSPHERE: cinematic lighting, dark mood, ...",
                )

                with gr.Row():
                    cn_negative = gr.Textbox(
                        label       = "❌ Negative Prompt",
                        lines       = 3,
                        interactive = True,
                        placeholder = "low quality, blurry, distorted, deformed, bad anatomy, watermark, text, ...",
                    )

                with gr.Row():
                    cn_save_btn = gr.Button("💾 Save controlnet.txt → workflow/", variant="secondary", scale=2)
                cn_status = gr.Textbox(label="Save Status", interactive=False, lines=1)

        # ── БЛОК: Проект-менеджер FLUX ────────────────────────────────
        gr.Markdown("---")
        with gr.Accordion("📁 Проект-менеджер (FLUX T2I промпты)", open=False):
            gr.Markdown(
                "Создаёт папку проекта и генерирует готовые **FLUX Text-to-Image промпты** для каждой сцены. "
                "Стиль определяется автоматически по выбранному нарратору. "
                "Используй эти промпты в FLUX → загружай готовые изображения в слоты → запускай LTX 2.3."
            )
            with gr.Row():
                create_project_btn = gr.Button(
                    "📁 Создать проект + FLUX-промпты",
                    variant = "primary",
                    scale   = 2,
                )
                project_seq_dd = gr.Dropdown(
                    label   = "Сцен в проекте",
                    choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    value   = 5,
                    scale   = 1,
                )

            project_path_out = gr.Textbox(
                label       = "📂 Путь к папке проекта",
                interactive = False,
                lines       = 1,
                placeholder = "Путь появится после создания...",
            )
            project_prompts_out = gr.Textbox(
                label       = "🖼️ Сгенерированные FLUX-промпты",
                lines       = 18,
                interactive = False,
                placeholder = "Промпты для каждой сцены появятся здесь...",
            )
            with gr.Row():
                project_copy_btn = gr.Button("📋 Copy промпты", scale=1, min_width=120)

        # ── Кнопки действий ───────────────────────────────────────────
        gr.Markdown("---")
        with gr.Row():
            gen_btn    = gr.Button("✍️ Сгенерировать",  variant="primary", scale=2)
            topics_btn = gr.Button("💡 10 тем",          scale=1)
            update_btn = gr.Button("🔄 Обновить базу",   scale=1)
            stats_btn  = gr.Button("📊 Статистика",      scale=1)
            cache_btn  = gr.Button("🧹 Очистить кэш",    scale=1)

        backend_lbl = gr.Textbox(label="Бэкенд", interactive=False, lines=1)

        with gr.Row():
            answer_box = gr.Textbox(
                label       = "Ответ",
                lines       = 20,
                interactive = False,
                scale       = 9,
            )
            copy_btn = gr.Button("📋 Copy", scale=1, min_width=60)

        with gr.Row():
            export_prompt_nophoto_btn = gr.Button("📝 Промт без фото", variant="secondary", scale=1)
        with gr.Row():
            export_btn = gr.Button("💾 Export (Bilder + Prompt)", variant="secondary", scale=2)
            json_btn   = gr.Button("🔄 JSON → Workflow", variant="secondary", scale=1)
        answer_export_status = gr.Textbox(label="Export Status", interactive=False, lines=2)

        # ── Перевод основного LTX-ответа (авто, если режим=сценарий) ─
        with gr.Group(visible=False) as answer_translation_col:
            gr.Markdown("---")
            gr.Markdown("#### 🇷🇺 Перевод промпта на русский")
            answer_ru_box = gr.Textbox(
                label       = "Перевод",
                lines       = 10,
                interactive = False,
            )
            answer_copy_ru_btn = gr.Button("📋 Copy перевод", min_width=120)

        with gr.Accordion("🔍 Найденные чанки", open=False):
            chunks_box = gr.Textbox(lines=15, interactive=False)

        update_log = gr.Textbox(label="Лог обновления базы", lines=8, interactive=False)

        # ── СОБЫТИЯ ───────────────────────────────────────────────────
        scan_btn.click(
            fn      = scan_models,
            inputs  = [model_folder],
            outputs = [model_dd, scan_status],
        )

        launch_btn.click(
            fn      = launch_llama,
            inputs  = [model_dd],
            outputs = [launch_log],
        ).then(
            fn      = backend_status,
            outputs = [status_box],
        )

        stop_btn.click(
            fn      = rag_core.stop_llama_server,
            outputs = [],
        ).then(
            fn      = backend_status,
            outputs = [status_box],
        )

        load_prompt_btn.click(
            fn      = load_system_prompt,
            inputs  = [prompt_path],
            outputs = [sys_prompt_box, prompt_load_status],
        )

        browse_btn.upload(
            fn      = lambda f: load_system_prompt(f.name),
            inputs  = [browse_btn],
            outputs = [sys_prompt_box, prompt_load_status],
        )

        copy_btn.click(
            fn      = None,
            inputs  = [answer_box],
            outputs = [],
            js      = "(text) => {if (navigator.clipboard && navigator.clipboard.writeText) {navigator.clipboard.writeText(text).catch(function() {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);});} else {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);}}",
        )

        # ── Export des generierten Prompts (mit/ohne Fotos) ────────────
        export_prompt_nophoto_btn.click(
            fn      = lambda p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, txt: export_answer_prompt(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, txt, with_photos=False),
            inputs  = img_slots + [answer_box],
            outputs = [answer_export_status],
        )

        all_cbs = [cb for row in cbs for cb in row]

        gen_btn.click(
            fn      = do_generate,
            inputs  = [question, style_dd, word_sl, mode_dd, lang_dd, script_dd,
                       narrator_dd, director_dd, n_seq_dd, cloud_dd, sys_prompt_box] + img_slots + all_cbs,
            outputs = [answer_box, answer_ru_box, chunks_box, backend_lbl, answer_translation_col],
        )

        answer_copy_ru_btn.click(
            fn      = None,
            inputs  = [answer_ru_box],
            outputs = [],
            js      = "(text) => {if (navigator.clipboard && navigator.clipboard.writeText) {navigator.clipboard.writeText(text).catch(function() {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);});} else {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);}}",
        )

        script_dd.change(
            fn      = lambda v: (
                gr.update(visible=(v == "сценарий")),
                gr.update(visible=(v == "сценарий")),
                gr.update(visible=(v == "сценарий")),
            ),
            inputs  = [script_dd],
            outputs = [narrator_dd, director_dd, timeline_column],
        )

        topics_btn.click(
            fn      = do_topics,
            inputs  = [question, cloud_dd],
            outputs = [answer_box, answer_ru_box, backend_lbl],
        )

        update_btn.click(
            fn      = do_update_db,
            outputs = [update_log],
        )

        stats_btn.click(
            fn      = do_stats,
            outputs = [answer_box],
        )

        cache_btn.click(
            fn      = rag_core.clear_cache,
            outputs = [answer_box],
        )

        # ── СОБЫТИЯ: динамические image-слоты ─────────────────────────
        tl_col_refs = [tl_col_1, tl_col_2, tl_col_3, tl_col_4, tl_col_5, tl_col_6, tl_col_7, tl_col_8, tl_col_9, tl_col_10]

        n_seq_dd.change(
            fn      = update_image_slots,
            inputs  = [n_seq_dd],
            outputs = img_slots + tl_col_refs + all_cbs,
        )

        # ── СОБЫТИЯ: Мульти-кадровый таймлайн ─────────────────────────
        def do_timeline(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, prov, n_frames, scenario):
            """Копирует изображения в стабильный кэш, затем вызывает Florence-2 + LLM."""
            all_paths = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
            stable_paths = [
                _safe_copy_image(p, i)
                for i, p in enumerate(all_paths)
            ]
            # Nur die aktiven Slots (gemäss n_frames) übergeben
            active_paths = stable_paths[:n_frames]
            try:
                return rag_core.generate_image_timeline(
                    image_paths    = active_paths,
                    cloud_provider = prov,
                    n_frames       = n_frames,
                    scenario       = (scenario or "").strip(),
                )
            except Exception as e:
                return f"[Ошибка при генерации таймлайна]: {e}"

        timeline_gen_btn.click(
            fn      = do_timeline,
            inputs  = img_slots + [cloud_dd, n_seq_dd, question],
            outputs = [timeline_out],
        )

        timeline_copy_btn.click(
            fn      = None,
            inputs  = [timeline_out],
            outputs = [],
            js      = "(text) => {if (navigator.clipboard && navigator.clipboard.writeText) {navigator.clipboard.writeText(text).catch(function() {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);});} else {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);}}",
        )

        export_btn.click(
            fn      = do_export,
            inputs  = img_slots + [timeline_out],
            outputs = [answer_export_status],
        )

        json_btn.click(
            fn      = do_json_workflow,
            inputs  = [],
            outputs = [answer_export_status],
        )

        translate_timeline_btn.click(
            fn      = lambda text, prov: rag_core.translate_to_russian(text, cloud_provider=prov),
            inputs  = [timeline_out, cloud_dd],
            outputs = [timeline_ru_out],
        )

        timeline_copy_ru_btn.click(
            fn      = None,
            inputs  = [timeline_ru_out],
            outputs = [],
            js      = "(text) => {if (navigator.clipboard && navigator.clipboard.writeText) {navigator.clipboard.writeText(text).catch(function() {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);});} else {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);}}",
        )

        # ── СОБЫТИЯ: Mode-Umschalter Video / ControlNet ────────────────
        img_mode_dd.change(
            fn      = lambda v: (
                gr.update(visible=(v == "Video")),
                gr.update(visible=(v == "Controlnet")),
            ),
            inputs  = [img_mode_dd],
            outputs = [video_col, controlnet_col],
        )

        # ── СОБЫТИЯ: ControlNet ────────────────────────────────────────
        cn_gen_btn.click(
            fn      = do_controlnet,
            inputs  = cn_img_slots + [cloud_dd],
            outputs = [cn_out],
        )

        cn_copy_btn.click(
            fn      = None,
            inputs  = [cn_out],
            outputs = [],
            js      = "(text) => {if (navigator.clipboard && navigator.clipboard.writeText) {navigator.clipboard.writeText(text).catch(function() {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);});} else {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);}}",
        )

        cn_save_btn.click(
            fn      = save_controlnet,
            inputs  = [cn_out, cn_negative],
            outputs = [cn_status],
        )

        # ── СОБЫТИЯ: Проект-менеджер ───────────────────────────────────
        create_project_btn.click(
            fn      = do_create_project,
            inputs  = [question, narrator_dd, project_seq_dd, cloud_dd],
            outputs = [project_path_out, project_prompts_out],
        )

        project_copy_btn.click(
            fn      = None,
            inputs  = [project_prompts_out],
            outputs = [],
            js      = "(text) => {if (navigator.clipboard && navigator.clipboard.writeText) {navigator.clipboard.writeText(text).catch(function() {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);});} else {var ta=document.createElement(\"textarea\");ta.value=text;ta.style.position=\"fixed\";ta.style.opacity=\"0\";document.body.appendChild(ta);ta.select();document.execCommand(\"copy\");document.body.removeChild(ta);}}",
        )

    return demo


# ──────────────────────────────────────────────
# ЗАПУСК
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("[RAG] Запускаю WebUI...")
    print(f"[RAG] Бэкенд: {backend_status()}")
    ui = build_ui()
    ui.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        inbrowser   = True,
    )
