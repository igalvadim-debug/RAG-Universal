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
SONGS_DIR    = BASE_DIR / "Songs"
SONGS_DIR.mkdir(exist_ok=True)

# workflow/tripple.py — создание workflow-JSON (TripleUpscale / CS LTX Director)
import sys as _sys
if str(BASE_DIR / "workflow") not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR / "workflow"))
import tripple  # noqa: E402
import grid_splitter  # noqa: E402  (Gitter/Storyboard in Einzelbilder schneiden)


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


def load_song_styles() -> list[tuple[str, str]]:
    """
    Liest styles_musik.txt (C:\Sandbox oder Projektordner) und gibt
    [(label, voller_Text)] für die 20 Musikstile zurück.
    Format der Datei: 'N. Titel\nBeschreibung\n\n...'
    """
    candidates = [
        Path(r"C:\Sandbox\styles_musik.txt"),
        BASE_DIR / "styles_musik.txt",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        result = []
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            label = lines[0].strip()
            if not label:
                continue
            result.append((label, block))
        if result:
            return result
    return []


_song_styles = load_song_styles()


def song_style_full_text(style_label: str) -> str:
    """Mappt das ausgewählte Stil-Label zurück auf den vollen Text aus styles_musik.txt."""
    if not style_label:
        return ""
    for label, full in _song_styles:
        if label == style_label:
            return full
    return style_label


def _song_essence(answer: str, question: str) -> str:
    """
    Zieht den 'Sinn' des Songs: zuerst die Titel-Zeile im SONG PROMPT, dann die erste
    Lyrik-Zeile nach [Strophe 1], sonst die Frage/Thema. Basis für den Dateinamen.
    """
    import re as _re
    # 1) Titel-Zeile im SONG PROMPT
    for line in answer.splitlines():
        line = line.strip()
        if line.lower().startswith(("titel:", "title:")):
            t = line.split(":", 1)[1].strip()
            if t and not t.startswith("["):
                return t[:70]
    # 2) erste Lyrik-Zeile nach [Strophe 1] / [Verse 1]
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if _re.match(r"^\s*\[(strophe|verse)\s*1\]", line, _re.IGNORECASE):
            for nxt in lines[i+1:i+5]:
                nxt = nxt.strip()
                if nxt and not nxt.startswith("[") and not nxt.lower().startswith(("strophe", "refrain", "chorus", "припев")):
                    return nxt[:70]
    # 3) Frage / Thema
    if question and question.strip():
        return question.strip()[:70]
    return "song"


def _slug_from_text(text: str, max_words: int = 4, max_len: int = 45) -> str:
    """Baut einen dateisicheren Slug aus den ersten Wörtern des Textes."""
    import re as _re
    _umlaut = {
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
    }
    words = []
    for w in text.strip().split():
        clean = _re.sub(r'[^a-zA-Z0-9\u0400-\u04FF]', '', w)
        for k, v in _umlaut.items():
            clean = clean.replace(k, v)
        if clean:
            words.append(clean.lower())
        if len(words) >= max_words:
            break
    slug = "_".join(words) if words else "song"
    return slug[:max_len]


def save_song_generation(answer: str, question: str, song_style: str, singer: str, language: str) -> str:
    """
    Speichert eine Song-Generation als TXT in D:\\claudeAgent\\RAG\\Songs,
    benannt nach dem Sinn des Songs (erste gesungene Zeile / Thema) + Zeitstempel.
    Gibt den Pfad zurück.
    """
    from datetime import datetime as _dt
    SONGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug_from_text(_song_essence(answer, question))
    ts   = _dt.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{slug}_{ts}.txt"
    path = SONGS_DIR / fname
    header = (
        f"# Song: {slug}\n"
        f"# Stil: {song_style}\n"
        f"# Sänger: {singer}\n"
        f"# Sprache: {language}\n"
        f"# Gespeichert: {_dt.now().isoformat(timespec='seconds')}\n"
        f"{'=' * 60}\n\n"
    )
    path.write_text(header + answer.strip() + "\n", encoding="utf-8")
    return str(path)


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
        port = rag_core.get_llama_port()
        return f"✅ llama-server (порт {port})" if port else "✅ llama-server"

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
                director_mode: str, n_sequences: int, video_length: int,
                song_style: str, singer: str, verses: int,
                cloud_provider: str, sys_prompt: str,
                img0, img1, img2, img3, img4, img5, img6, img7, img8, img9,
                ref0, ref1, ref2,
                cb_0_0, cb_0_1, cb_0_2, cb_0_3, cb_0_4, cb_0_5, cb_0_6, cb_0_7, cb_0_8, cb_0_9,
                cb_1_0, cb_1_1, cb_1_2, cb_1_3, cb_1_4, cb_1_5, cb_1_6, cb_1_7, cb_1_8, cb_1_9,
                cb_2_0, cb_2_1, cb_2_2, cb_2_3, cb_2_4, cb_2_5, cb_2_6, cb_2_7, cb_2_8, cb_2_9,
                cb_3_0, cb_3_1, cb_3_2, cb_3_3, cb_3_4, cb_3_5, cb_3_6, cb_3_7, cb_3_8, cb_3_9,
                cb_4_0, cb_4_1, cb_4_2, cb_4_3, cb_4_4, cb_4_5, cb_4_6, cb_4_7, cb_4_8, cb_4_9,
                cb_5_0, cb_5_1, cb_5_2, cb_5_3, cb_5_4, cb_5_5, cb_5_6, cb_5_7, cb_5_8, cb_5_9,
                cb_6_0, cb_6_1, cb_6_2, cb_6_3, cb_6_4, cb_6_5, cb_6_6, cb_6_7, cb_6_8, cb_6_9,
                cb_7_0, cb_7_1, cb_7_2, cb_7_3, cb_7_4, cb_7_5, cb_7_6, cb_7_7, cb_7_8, cb_7_9,
                cb_8_0, cb_8_1, cb_8_2, cb_8_3, cb_8_4, cb_8_5, cb_8_6, cb_8_7, cb_8_8, cb_8_9,
                cb_9_0, cb_9_1, cb_9_2, cb_9_3, cb_9_4, cb_9_5, cb_9_6, cb_9_7, cb_9_8, cb_9_9,
                cb_10_0, cb_10_1, cb_10_2, cb_10_3, cb_10_4, cb_10_5, cb_10_6, cb_10_7, cb_10_8, cb_10_9,
                cb_11_0, cb_11_1, cb_11_2, cb_11_3, cb_11_4, cb_11_5, cb_11_6, cb_11_7, cb_11_8, cb_11_9,
                cb_12_0, cb_12_1, cb_12_2, cb_12_3, cb_12_4, cb_12_5, cb_12_6, cb_12_7, cb_12_8, cb_12_9,
                cb_13_0, cb_13_1, cb_13_2, cb_13_3, cb_13_4, cb_13_5, cb_13_6, cb_13_7, cb_13_8, cb_13_9,
                cb_14_0, cb_14_1, cb_14_2, cb_14_3, cb_14_4, cb_14_5, cb_14_6, cb_14_7, cb_14_8, cb_14_9,
                cb_15_0, cb_15_1, cb_15_2, cb_15_3, cb_15_4, cb_15_5, cb_15_6, cb_15_7, cb_15_8, cb_15_9,
                cb_16_0, cb_16_1, cb_16_2, cb_16_3, cb_16_4, cb_16_5, cb_16_6, cb_16_7, cb_16_8, cb_16_9,
                cb_17_0, cb_17_1, cb_17_2, cb_17_3, cb_17_4, cb_17_5, cb_17_6, cb_17_7, cb_17_8, cb_17_9,
                cb_18_0, cb_18_1, cb_18_2, cb_18_3, cb_18_4, cb_18_5, cb_18_6, cb_18_7, cb_18_8, cb_18_9,
                cb_19_0, cb_19_1, cb_19_2, cb_19_3, cb_19_4, cb_19_5, cb_19_6, cb_19_7, cb_19_8, cb_19_9,
                rag_collection: str = "", cs_timelines: int = 1, n_seq_tl2: int = 0,
                ref2_0=None, ref2_1=None, ref2_2=None,
                t2_0=None, t2_1=None, t2_2=None, t2_3=None, t2_4=None,
                t2_5=None, t2_6=None, t2_7=None, t2_8=None, t2_9=None):
    if not question.strip():
        return "⚠️ Введи вопрос или тему.", "", "", backend_status(), gr.update(visible=False)

    if script_mode in ("сценарий", "создать песню"):
        eff_script_mode = script_mode
    elif mode == "аналитический":
        eff_script_mode = "аналитический"
    else:
        eff_script_mode = "художественный"
    if eff_script_mode == "создать песню":
        director_mode = "нет"   # Song-Modus nutzt das SONG-PROMPT-Format, nicht den Director Mode

    # ── Сценарий-зависимые картинки: TripleUpscale = 1 стартовый кадр I2V, ──
    #    CS LTX Director = @ref1-3 на таймлайн (VL-LLM) + обычные таймлайн-слоты (или без картинок)
    if director_mode == "TripleUpscale":
        image_paths = [img0]
        ref_image_paths = None
        ref_image_paths_tl2 = None
        n_sequences_tl2 = 0
    elif director_mode in ("Minimax H3", "Minimax H3 Anim"):
        # до 9 референс-картинок = <Picture 1..9> (порядок слотов, пустые пропускаются)
        image_paths = [img0, img1, img2, img3, img4, img5, img6, img7, img8]
        ref_image_paths = None
        ref_image_paths_tl2 = None
        n_sequences_tl2 = 0
    elif director_mode == "LTX 2.5 T2V":
        # чистый текст — картинок нет
        image_paths = None
        ref_image_paths = None
        ref_image_paths_tl2 = None
        n_sequences_tl2 = 0
    elif director_mode == "LTX 2.5 First/Last":
        # первый + последний кадр (START/END) — не отдельные сцены
        image_paths = [img0, img1]
        ref_image_paths = None
        ref_image_paths_tl2 = None
        n_sequences_tl2 = 0
    elif director_mode == "LTX 2.5 Director":
        # как CS LTX Director, но ОДИН таймлайн (нода LTXDirectorCS25)
        image_paths = [img0, img1, img2, img3, img4, img5, img6, img7, img8, img9]
        ref_image_paths = [_norm_files(ref0), _norm_files(ref1), _norm_files(ref2)]
        ref_image_paths_tl2 = None
        n_sequences_tl2 = 0
    elif director_mode == "CS LTX Director":
        two_tl = (int(cs_timelines or 1) == 2)
        image_paths = ([img0, img1, img2, img3, img4, img5, img6, img7, img8, img9]
                       + ([t2_0, t2_1, t2_2, t2_3, t2_4, t2_5, t2_6, t2_7, t2_8, t2_9] if two_tl else []))
        ref_image_paths = [_norm_files(ref0), _norm_files(ref1), _norm_files(ref2)]
        ref_image_paths_tl2 = [_norm_files(ref2_0), _norm_files(ref2_1), _norm_files(ref2_2)] if two_tl else None
        n_sequences_tl2 = int(n_seq_tl2) if two_tl else 0
    else:
        image_paths = [img0, img1, img2, img3, img4, img5, img6, img7, img8, img9]
        ref_image_paths = None
        ref_image_paths_tl2 = None
        n_sequences_tl2 = 0

    technique_names = ["Dutch Angle", "Dolly Zoom", "Push-In", "Low-Angle", "POV", "Rack Focus", "Static Long Take", "Whip Pan", "Snap Zoom", "Overhead Shot",
                       "Eye Level", "High Angle", "Bird's Eye View", "Worm's Eye View", "Over-the-Shoulder", "Establishing Shot", "Wide Shot", "Close-Up", "Two Shot", "Reaction Shot"]
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
        [cb_10_0, cb_10_1, cb_10_2, cb_10_3, cb_10_4, cb_10_5, cb_10_6, cb_10_7, cb_10_8, cb_10_9],
        [cb_11_0, cb_11_1, cb_11_2, cb_11_3, cb_11_4, cb_11_5, cb_11_6, cb_11_7, cb_11_8, cb_11_9],
        [cb_12_0, cb_12_1, cb_12_2, cb_12_3, cb_12_4, cb_12_5, cb_12_6, cb_12_7, cb_12_8, cb_12_9],
        [cb_13_0, cb_13_1, cb_13_2, cb_13_3, cb_13_4, cb_13_5, cb_13_6, cb_13_7, cb_13_8, cb_13_9],
        [cb_14_0, cb_14_1, cb_14_2, cb_14_3, cb_14_4, cb_14_5, cb_14_6, cb_14_7, cb_14_8, cb_14_9],
        [cb_15_0, cb_15_1, cb_15_2, cb_15_3, cb_15_4, cb_15_5, cb_15_6, cb_15_7, cb_15_8, cb_15_9],
        [cb_16_0, cb_16_1, cb_16_2, cb_16_3, cb_16_4, cb_16_5, cb_16_6, cb_16_7, cb_16_8, cb_16_9],
        [cb_17_0, cb_17_1, cb_17_2, cb_17_3, cb_17_4, cb_17_5, cb_17_6, cb_17_7, cb_17_8, cb_17_9],
        [cb_18_0, cb_18_1, cb_18_2, cb_18_3, cb_18_4, cb_18_5, cb_18_6, cb_18_7, cb_18_8, cb_18_9],
        [cb_19_0, cb_19_1, cb_19_2, cb_19_3, cb_19_4, cb_19_5, cb_19_6, cb_19_7, cb_19_8, cb_19_9],
    ]
    timeline_techniques = []
    for c in range(10):
        selected = None
        for r in range(20):
            if cb_grid[r][c]:
                selected = technique_names[r]
                break
        timeline_techniques.append(selected)

    # RAG-Suche — darf die Generierung niemals crashen (Embedding-Modell/Chroma-Fehler → ohne Kontext weiter)
    try:
        chunks  = rag_core.search_chunks(question, collection_name=rag_collection)
        context = rag_core.build_context(chunks)
    except Exception as e:
        rag_core._debug_log(f"⚠️ RAG-Suche fehlgeschlagen: {e} — generiere ohne Kontext.")
        chunks  = []
        context = ""
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
        video_length        = int(video_length) if video_length else 30,
        song_verses         = int(verses) if verses else 2,
        singer              = singer or "женщина",
        music_style         = song_style_full_text(song_style),
        image_paths         = image_paths,
        ref_image_paths     = ref_image_paths,
        ref_image_paths_tl2 = ref_image_paths_tl2,
        n_sequences_tl2     = n_sequences_tl2,
    )
    rag_core._debug_log(f"Generierung fertig: {len(answer)} Zeichen | Modus={eff_script_mode} | Anfang: {answer[:80]!r}")

    # Перевод на русский если режим сценарий (НЕ для музыкальных клипов — песни bleiben in Originalsprache)
    translation = ""
    is_music = narrator.startswith("music:")
    if eff_script_mode == "сценарий" and answer.strip() and not is_music:
        translation = rag_core.translate_to_russian(answer, cloud_provider=cloud_provider)

    global _last_chunks
    _last_chunks = chunks
    chunks_text  = "\n\n" + ("─" * 60) + "\n\n".join(chunks)

    # Song-/Musik-Generation automatisch als TXT speichern (benannt nach dem Sinn des Songs)
    if (eff_script_mode == "создать песню" or is_music) and answer.strip() and not answer.strip().startswith("[Ошибка]"):
        try:
            saved_path = save_song_generation(answer, question, song_style, singer, language)
            chunks_text += f"\n\n✅ Song gespeichert: {saved_path}"
            rag_core._debug_log(f"✅ Song gespeichert: {saved_path}")
        except Exception as e:
            chunks_text += f"\n\n⚠️ Song-Speicherung fehlgeschlagen: {e}"
            rag_core._debug_log(f"⚠️ Song-Speicherung fehlgeschlagen: {e}")

    return answer, translation, chunks_text, backend_status(), gr.update(visible=bool(translation.strip()))


def do_topics(seed: str, cloud_provider: str = "auto", collection_name: str = ""):
    result = rag_core.generate_topics(seed=seed, cloud_provider=cloud_provider, collection_name=collection_name)
    return result, "", backend_status()


def do_update_db():
    """Index new Markdown documents and return newline-joined update messages.

    Run the updater twice if the first run and cache cleanup succeed.
    Clear the search and health caches and unload the cached embedding model
    after the first attempt, even if it fails. Clear the search cache again
    after the second run succeeds.

    Errors raised by the updater or cache cleanup propagate instead of
    returning messages; a cleanup error can supersede an indexing error.
    """
    lines = []
    def cb(msg):
        lines.append(msg)
    try:
        update_db.main(log_callback=cb)
    finally:
        rag_core.clear_cache()
    update_db.main(log_callback=cb)
    rag_core.clear_search_cache()  # база изменилась — кэш поиска неактуален
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


def _rand_folder(base_dir: Path, prefix: str) -> Path:
    """Папка <prefix>_<3 случайных символа> в base_dir (защита от коллизий)."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
        folder = base_dir / f"{prefix}_{rand}"
        if not folder.exists():
            break
    folder.mkdir(parents=True, exist_ok=True)
    return folder


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


def _split_and_save_notes(content: str, proj_dir: Path) -> str:
    """
    Trennt einen evtl. angehaengten ===EMPFEHLUNGEN=== Block (LLM-Meta-Kommentare,
    von rag_core.py / image_captioning.py erzeugt) vom eigentlichen Prompt-Text ab
    und speichert ihn separat als empfehlungen.txt im Projektordner, statt
    video.txt/novideo.txt zu verunreinigen. Gibt den bereinigten Haupttext zurueck.
    """
    marker = "===EMPFEHLUNGEN==="
    if marker in content:
        main_part, notes_part = content.split(marker, 1)
        main_part = main_part.rstrip()
        notes_part = notes_part.strip()
        if notes_part:
            (proj_dir / "empfehlungen.txt").write_text(notes_part, encoding="utf-8")
        return main_part
    return content


def _rand_folder(base_dir: Path, prefix: str) -> Path:
    """Папка <prefix>_<3 случайных символа> в base_dir (защита от коллизий)."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
        folder = base_dir / f"{prefix}_{rand}"
        if not folder.exists():
            break
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def do_export(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, prompt_text):
    """
    Exportiert Bilder + Prompt:
    • ComfyUiVid\\input\\ — Bilder (a_XXXX_.png, ...)
    • workflow\\video_<3 Zeichen>\\ — Bilder + video.txt/novideo.txt
    """
    comfy_dir = Path(r"D:\ComfyUiVid\input")
    comfy_dir.mkdir(parents=True, exist_ok=True)

    content = (prompt_text or "").strip()
    images  = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    slot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    letters = string.ascii_lowercase

    # Projektordner: короткое имя video_<3>
    proj_dir = _rand_folder(Path(r"D:\claudeAgent\RAG\workflow"), "video") if content else Path(r"D:\claudeAgent\RAG\workflow")
    content = _split_and_save_notes(content, proj_dir)

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

    pic_ok = _maybe_write_picture_prompts(proj_dir, content, images)

    if exported > 0:
        return f"✅ {exported} Bilder + {txt_name} → {proj_dir.name} & ComfyUiVid\\input" + (" + picture_prompts.txt" if pic_ok else "")
    else:
        return f"⚠️ Keine Bilder — {txt_name} → {proj_dir.name}" + (" + picture_prompts.txt" if pic_ok else "")


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
    with_photos=True  → ComfyUiVid\\input + workflow\\video_<3 Zeichen>\\ (Bilder + video.txt)
    with_photos=False → ComfyUiVid\\input + workflow\\novid_<3 Zeichen>\\ (nur novideo.txt)
    """
    comfy_dir = Path(r"D:\ComfyUiVid\input")
    comfy_dir.mkdir(parents=True, exist_ok=True)

    images  = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    slot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    letters = string.ascii_lowercase
    content = (prompt_text or "").strip()

    if not content:
        return "⚠️ Kein Prompt zum Exportieren — erst generieren."

    # ── папка: с фото — workflow/video_<3>, без фото — workflow/novid_<3> ──
    if with_photos:
        proj_dir = _rand_folder(Path(r"D:\claudeAgent\RAG\workflow"), "video")
    else:
        proj_dir = _rand_folder(Path(r"D:\claudeAgent\RAG\workflow"), "novid")

    content = _split_and_save_notes(content, proj_dir)

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

        pic_ok = _maybe_write_picture_prompts(proj_dir, content, images)
        _suffix = " + picture_prompts.txt" if pic_ok else ""
        if exported > 0:
            return f"✅ Промт с фото: {exported} Bilder + {txt_name} → {proj_dir.name} & ComfyUiVid\\input{_suffix}"
        else:
            return f"⚠️ Keine Bilder — {txt_name} → {proj_dir.name} (nur Prompt){_suffix}"
    else:
        txt_name = "novideo.txt"
        (proj_dir / txt_name).write_text(content, encoding="utf-8")
        (comfy_dir / txt_name).write_text(content, encoding="utf-8")
        pic_ok = _maybe_write_picture_prompts(proj_dir, content, images)
        _suffix = " + picture_prompts.txt" if pic_ok else ""
        return f"✅ Промт ohne Fotos: {txt_name} → {proj_dir.name} & ComfyUiVid\\input ({len(content)} Zeichen){_suffix}"


def _comfy_input_dir() -> Path:
    return Path(r"D:\ComfyUiVid\input")


def _norm_files(v, max_files: int = 2):
    """Нормализует значение gr.File(file_count='multiple'): → список путей (str).
    Максимум max_files картинок на слот (по умолчанию 2)."""
    if not v:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.append(x.get("path") or x.get("name") or "")
        return [x for x in out if x][:max_files]
    if isinstance(v, dict):
        return [v.get("path") or v.get("name") or ""][:max_files]
    return [v][:max_files]


def export_tripple_json(answer_text, img0, video_length, director_mode):
    """
    Собирает mod_Tripple.json из ответа TripleUpscale:
    новая папка workflow/tripple_cxnc_<random> + единственная картинка (I2V).
    JSON создаёт tripple.py (принцип update_workflow.py).
    """
    if director_mode != "TripleUpscale":
        return "⚠️ Сначала выбери сценарий «TripleUpscale» и нажми «Сгенерировать»."

    prompts = tripple.parse_tripple_answer(answer_text or "")
    if not prompts.get("scene_prompt") or not prompts.get("global_positive"):
        return ("⚠️ В ответе нет формата TripleUpscale (---GLOBAL_POSITIVE_START--- / ---SCENE_PROMPT_START---). "
                "Сгенерируй промт в режиме TripleUpscale.")

    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    # папка triple_<3 случайных символа> (короткое имя)
    folder = _rand_folder(workflow_dir, "triple")

    # единственная картинка: в папку проекта + в ComfyUI input
    img_file = None
    if img0:
        src = Path(img0)
        if src.exists():
            img_file = src.name
            shutil.copy2(src, folder / img_file)

    try:
        frames = tripple.frames_for_seconds(int(video_length) if video_length else 20)
        out = tripple.create_tripple_workflow(
            template_path    = workflow_dir / "Tripple.json",
            out_path         = folder / "mod_Tripple.json",
            prompts          = prompts,
            image_path       = img0,
            fps              = 24,
            frames           = frames,
            comfy_input_dir  = _comfy_input_dir(),
        )
    except Exception as e:
        return f"❌ Ошибка создания Tripple-JSON: {e}"

    (folder / "video.txt").write_text((answer_text or "").strip(), encoding="utf-8")

    # промт стартового кадра для FLUX 2 Klein — отдельным txt (для T2I-генерации картинки)
    flux_txt = ""
    flux_prompt = (prompts.get("flux_start_image") or "").strip()
    if flux_prompt:
        flux_txt = "flux_start_image.txt"
        (folder / flux_txt).write_text(flux_prompt + "\n", encoding="utf-8")

    parts = [f"✅ Tripple-JSON: {out.name} → {folder.name}",
             f"{frames} кадров @ 24 fps",
             f"стартовый кадр: {img_file or '—'}"]
    if flux_txt:
        parts.append(f"FLUX-промпт: {flux_txt}")
    return " | ".join(parts)


def export_ltx25_json(answer_text, img_first, img_last, video_length, director_mode):
    """
    Собирает mod_ltx25.json из ответа LTX 2.5 First/Last:
    папка ltx25_<3>, промт в PrimitiveStringMultiline, первый/последний кадр в LoadImage,
    Duration (сек) из UI, ManualSigmas — шаги по SIGMA_STEPS от LLM (или от длительности).
    """
    if director_mode != "LTX 2.5 First/Last":
        return "⚠️ Сначала выбери сценарий «LTX 2.5 First/Last» и нажми «Сгенерировать»."

    content = (answer_text or "").strip()
    if not content:
        return "⚠️ Нет промта — сгенерируй его в режиме LTX 2.5 First/Last."
    prompt_clean, sigma_steps = tripple.parse_sigma_steps(content)
    if not prompt_clean:
        return "⚠️ Промт пуст после удаления SIGMA_STEPS."

    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    folder = _rand_folder(workflow_dir, "ltx25")
    duration = int(video_length) if video_length else 15
    sig = tripple.compute_sigmas(duration, 24, steps=sigma_steps)

    try:
        out = tripple.create_ltx25_workflow(
            template_path   = workflow_dir / "LTX 2.5 First and Last Frame Video Generation .json",
            out_path        = folder / "mod_ltx25.json",
            prompt_text     = prompt_clean,
            first_image     = img_first,
            last_image      = img_last,
            duration        = duration,
            fps             = 24,
            sigma_values    = sig,
            comfy_input_dir = _comfy_input_dir(),
        )
    except Exception as e:
        return f"❌ Ошибка создания LTX-2.5 JSON: {e}"

    (folder / "video.txt").write_text(prompt_clean + "\n", encoding="utf-8")
    f1 = Path(img_first).name if img_first else "—"
    f2 = Path(img_last).name if img_last else "—"
    src = f"{len(sig)} шагов (решение LLM)" if sigma_steps else f"{len(sig)} шагов (по длительности)"
    return (f"✅ LTX-2.5-JSON: {out.name} → {folder.name} | "
            f"{duration}с, сигамы: {src} | кадры: {f1} → {f2}")


def export_minimax_json(answer_text, video_length, director_mode, *imgs):
    """
    MiniMax H3 Reference-to-Video: mod_minimax_ref.json в новой папке mmref_<3> / mmanim_<3>.
    Ответ LLM (VIDEO/SOUND/MUSIC) объединяется в одну строку промта, картинки из слотов
    (порядок слотов = <Picture N>) копируются в ComfyUI input, Duration — из слайдера.
    """
    if director_mode not in ("Minimax H3", "Minimax H3 Anim"):
        return "⚠️ Сначала выбери сценарий «Minimax H3» или «Minimax H3 Anim» и нажми «Сгенерировать»."

    content = (answer_text or "").strip()
    if not content:
        return "⚠️ Нет промта — сгенерируй его в режиме Minimax H3."
    merged, _parts = tripple.parse_minimax_answer(content)
    if not merged:
        return "⚠️ Промт пуст после разбора VIDEO/SOUND/MUSIC."

    pics = [p for p in imgs[:9] if p]
    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    folder = _rand_folder(workflow_dir, "mmanim" if director_mode == "Minimax H3 Anim" else "mmref")
    duration = int(video_length) if video_length else 10

    try:
        out = tripple.create_minimax_ref_workflow(
            template_path   = workflow_dir / "Minimax_reference_pictures.json",
            out_path        = folder / "mod_minimax_ref.json",
            prompt_text     = merged,
            images          = pics,
            duration        = duration,
            comfy_input_dir = _comfy_input_dir(),
        )
    except Exception as e:
        return f"❌ Ошибка создания MiniMax-JSON: {e}"

    (folder / "video.txt").write_text(content + "\n", encoding="utf-8")
    note = " | ⚠️ H3 лучше держит 5–15 с" if duration > 15 else ""
    return (f"✅ MiniMax-JSON: {out.name} → {folder.name} | {duration}с, "
            f"картинок: {len(pics)} (<Picture 1…{len(pics)}>){note}")


def export_t2v_json(answer_text, video_length, director_mode):
    """
    Text-to-Video (Fast LTX2.5): mod_t2v.json в папке t2v_<3>.
    Промт в PrimitiveStringMultiline, Duration (сек) из UI, ManualSigmas ×2 (low/high).
    """
    if director_mode != "LTX 2.5 T2V":
        return "⚠️ Сначала выбери сценарий «LTX 2.5 T2V» и нажми «Сгенерировать»."

    content = (answer_text or "").strip()
    if not content:
        return "⚠️ Нет промта — сгенерируй его в режиме LTX 2.5 T2V."
    prompt_clean, sigma_steps = tripple.parse_sigma_steps(content)
    if not prompt_clean:
        return "⚠️ Промт пуст после удаления SIGMA_STEPS."

    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    folder = _rand_folder(workflow_dir, "t2v")
    duration = int(video_length) if video_length else 16
    sl, sh = tripple.compute_t2v_sigmas(duration, 24, high_steps=sigma_steps)

    try:
        out = tripple.create_t2v_workflow(
            template_path = workflow_dir / "Text-to-Video Fast LTX2.5 Sync Audio & Video.json",
            out_path      = folder / "mod_t2v.json",
            prompt_text   = prompt_clean,
            duration      = duration,
            fps           = 24,
            sigmas_low    = sl,
            sigmas_high   = sh,
        )
    except Exception as e:
        return f"❌ Ошибка создания T2V JSON: {e}"

    (folder / "video.txt").write_text(prompt_clean + "\n", encoding="utf-8")
    src = f"решение LLM" if sigma_steps else f"по длительности"
    return f"✅ T2V-JSON: {out.name} → {folder.name} | {duration}с, сигамы low {len(sl)} / high {len(sh)} шагов ({src})"


def _caption_ref_images(ref_paths: list, timeline: int = 1):
    """
    VL-LLM описания @ref-картинок для CS LTX Director (таймлайн 1 или 2):
    свежие из rag_core._last_ref_descriptions (последняя генерация, с ключом timeline),
    иначе QwenVL-llama-server / Florence-2 (image_captioning.py).
    Возвращает {path: description}.
    """
    fresh = {d.get("image"): d.get("description", "")
             for d in (getattr(rag_core, "_last_ref_descriptions", None) or [])
             if int(d.get("timeline", 1)) == timeline}
    result = {}
    missing = []
    for p in ref_paths:
        if not p:
            continue
        fname = Path(p).name
        if fname in fresh and fresh[fname]:
            result[p] = fresh[fname]
        else:
            missing.append(p)
    if missing:
        try:
            from image_captioning import QwenVLCaptioner, Florence2Captioner
            captioner = None
            try:
                captioner = QwenVLCaptioner()
                captioner.caption_anchor(missing[0])
            except Exception:
                captioner = Florence2Captioner()
            try:
                for p in missing:
                    result[p] = captioner.caption_anchor(p)
            finally:
                captioner.unload()
        except Exception as e:
            for p in missing:
                result[p] = f"[описание недоступно: {e}]"
    return result


def _generate_picture_prompts(scenes: list, with_anchors: bool = True):
    """
    Рекомендации picture-промтов (Flux 2 Klein) для сцен:
      - SCENE_N_PICTURE — кадр сцены целиком (только если картинка сцены НЕ загружена)
      - SCENE_N_ANCHOR  — анкор начала сцены (только при with_anchors=True, CS Director:
        ключевой кадр фиксирует идентичность персонажа/объекта/окружения на старте)
    scenes: [(label, prompt, has_image, refs), ...] — refs = [desc1, desc2, desc3]
    (описания @ref1-3 таймлайна; теги @refN подставляются описаниями, в Flux они не нужны).
    label вида "SCENE 3 (Timeline 1)".
    Генерация ЧАНКАМИ по 3 сцены + повтор пропущенных — чтобы LLM не обрывался
    на середине вывода (обрыв на одном большом запросе).
    Возвращает текст с маркерами (готов к записи в txt) или "" при ошибке.
    """
    import re as _re

    # Промт уже готов — RAG/эмбеддинги не нужны: выгружаем модель, освобождая VRAM под LLM.
    try:
        rag_core.clear_cache()
    except Exception:
        pass

    fmt = (
        "For every requested scene output EXACTLY this parseable format:\n"
        "---SCENE_N_PICTURE_START---\n"
        "[picture prompt of the whole scene as a STILL FRAME — only if the scene's image is NOT loaded; "
        "otherwise write NONE]\n"
        "---SCENE_N_PICTURE_END---\n"
    )
    if with_anchors:
        fmt += (
            "---SCENE_N_ANCHOR_START---\n"
            "[picture prompt for the ANCHOR image at the START of this scene — the keyframe that pins the "
            "visual identity (character face, vehicle, environment, lighting) so it stays identical when the "
            "video begins; always write it]\n"
            "---SCENE_N_ANCHOR_END---\n"
        )
    system = (
        "You convert VIDEO scene prompts into FLUX 2 KLEIN picture prompts (text-to-image).\n"
        "Plain natural language, 50-150 words each, structure: [Subject] in [Setting/Environment], "
        "[Style/Medium], [Lighting], [Additional Details]. NO quality tags or keyword stacks "
        "('masterpiece', '8k', 'best quality').\n"
        + fmt +
        "Rules: use the scene's concrete visible details (subject, environment, light, camera framing as a still); "
        "NEVER output @ref1/@ref2/@ref3 tags in the picture or anchor prompts — the referenced subjects are "
        "already described in the scene text; write their concrete appearance instead. "
        "English only. No meta commentary outside the markers."
    )
    if with_anchors:
        system += " The anchor focuses on what must stay IDENTICAL at the scene start."

    _block_re = _re.compile(
        r"---SCENE[_\s]*(\d+)[_\s]*(PICTURE|ANCHOR)_START---\s*(.*?)\s*---SCENE[_\s]*\d+[_\s]*\2_END---",
        _re.DOTALL,
    )

    def _extract(answer):
        got = {}
        for m in _block_re.finditer(answer or ""):
            got[(int(m.group(1)), m.group(2))] = m.group(3).strip()
        return got

    def _num(label):
        m = _re.search(r"SCENE\s*(\d+)", label)
        return int(m.group(1)) if m else 0

    def _subst(text, refs):
        """@refN → описание референса (в Flux теги бесполезны)."""
        for i, desc in enumerate(refs or [], 1):
            if desc:
                text = text.replace(f"@ref{i}", desc.strip())
        return text

    def _call(chunk):
        user = "\n\n".join(
            f"{label} (image loaded: {'no' if not has_img else 'yes'}):\n{_subst(text, refs)}"
            for label, text, has_img, refs in chunk
        )
        ans = rag_core.call_llm(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=6000,
            temperature=0.7,
        )
        return (ans or "").strip()

    blocks = []
    CHUNK = 3
    for start in range(0, len(scenes), CHUNK):
        chunk = scenes[start:start + CHUNK]
        got = _extract(_call(chunk))

        # повтор пропущенных сцен (обрыв модели) — до 3 попыток на чанк
        for _attempt in range(3):
            missing_nums = []
            for label, _t, _h, _r in chunk:
                num = _num(label)
                for kind in (("PICTURE",) if not with_anchors else ("PICTURE", "ANCHOR")):
                    if (num, kind) not in got:
                        missing_nums.append(num)
            missing_nums = list(dict.fromkeys(missing_nums))
            if not missing_nums:
                break
            retry = []
            for num in missing_nums:
                src = next((t for l, t, h, r in chunk if _num(l) == num), "")
                has_img = next((h for l, t, h, r in chunk if _num(l) == num), False)
                refs = next((r for l, t, h, r in chunk if _num(l) == num), [])
                retry.append((f"SCENE {num} (retry)", src, has_img, refs))
            got.update(_extract(_call(retry)))

        # собираем в порядке чанка
        for label, _t, _h, _r in chunk:
            num = _num(label)
            pic = got.get((num, "PICTURE"))
            if pic:
                pic = _clean_refs(pic)
                blocks.append(f"---SCENE_{num}_PICTURE_START---\n{pic}\n---SCENE_{num}_PICTURE_END---")
            if with_anchors:
                anc = got.get((num, "ANCHOR"))
                if anc:
                    anc = _clean_refs(anc)
                    blocks.append(f"---SCENE_{num}_ANCHOR_START---\n{anc}\n---SCENE_{num}_ANCHOR_END---")

    return "\n".join(blocks)


def _clean_refs(text):
    """Страховка: убирает любые @refN из picture/anchor промтов (Flux их не понимает)."""
    import re as _re
    return _re.sub(r"\s*@ref\d+\s*", " ", text).replace("  ", " ").strip()


def _maybe_write_picture_prompts(proj_dir, content, images):
    """Для классического Director: picture-промты сцен БЕЗ анкеров (если картинки пропущены)."""
    try:
        _gp, seqs, _durs, _wts = tripple.parse_director_answer(content or "")
        if not seqs:
            return False  # не Director/CS-формат (Tripple и т.п.) — пропускаем
        scenes = []
        for i, s in enumerate(seqs, 1):
            scenes.append((f"SCENE {i}", s, bool(images[i - 1] if i - 1 < len(images) else None), ["", "", ""]))
        pic_txt = _generate_picture_prompts(scenes, with_anchors=False)
        if pic_txt:
            (proj_dir / "picture_prompts.txt").write_text(pic_txt + "\n", encoding="utf-8")
            return True
    except Exception as e:
        rag_core._debug_log(f"⚠️ picture_prompts.txt не создан: {e}")
    return False


def export_cs_json(answer_text, n_seq, n_seq_tl2, video_length, director_mode, prov,
                   cs_timelines,
                   ref0, ref1, ref2, ref2_0, ref2_1, ref2_2,
                   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                   t2_0, t2_1, t2_2, t2_3, t2_4, t2_5, t2_6, t2_7, t2_8, t2_9):
    """
    Собирает mod_SC.json из ответа CS LTX Director (принцип high.json):
    единый global_prompt + ---SEQUENCE--- сегменты; при 2 таймлайнах результат режется
    по индексу: первые N1 SEQUENCE → таймлайн 1 (нода LTXDirectorCS слева), следующие N2 → таймлайн 2.
    Длительность делится пропорционально N1/(N1+N2). JSON создаёт tripple.py (timelines=[...]).
    """
    if director_mode != "CS LTX Director":
        return "⚠️ Сначала выбери сценарий «CS LTX Director» и нажми «Сгенерировать»."

    global_prompt, seqs, durs, weights = tripple.parse_director_answer(answer_text or "")
    if not seqs:
        return ("⚠️ В ответе нет формата CS/Директор (---GLOBAL_PROMPT_START--- / ---SEQUENCE_N_START---). "
                "Сгенерируй промт в режиме CS LTX Director.")

    two = (int(cs_timelines or 1) == 2)
    n1 = max(int(n_seq) if n_seq else 5, 1)
    n2 = max(int(n_seq_tl2) if n_seq_tl2 else 0, 0) if two else 0

    # ── разрез по индексу: первые N1 SEQUENCE → TL1, следующие N2 → TL2 ──
    tl1_seqs = seqs[:n1]
    tl2_seqs = seqs[n1:n1 + n2] if two else []
    tl1_durs = durs[:n1]
    tl2_durs = durs[n1:n1 + n2] if two else []
    tl1_wts = weights[:n1]
    tl2_wts = weights[n1:n1 + n2] if two else []
    if not tl1_seqs:
        return "⚠️ Нет SEQUENCE для таймлайна 1 — сгенерируй промт с количеством сцен ≥ N1."
    if two and not tl2_seqs:
        return (f"⚠️ В ответе только {len(seqs)} сцен, а нужно ≥ {n1 + n2} (N1+N2) — "
                f"сгенерируй промт заново в режиме CS LTX Director.")

    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    comfy = _comfy_input_dir()

    # ── длительность пропорционально: N1/(N1+N2) → TL1, остаток → TL2 ──
    frames_total = tripple.frames_for_seconds(int(video_length) if video_length else 30)
    if two and (n1 + n2) > 0:
        tl1_frames = max(1, round(frames_total * n1 / (n1 + n2)))
        tl2_frames = frames_total - tl1_frames
    else:
        tl1_frames = frames_total
        tl2_frames = 0

    # папка проекта workflow/Workflow_<3 случайных символа>; подпапки ComfyUI раздельные на таймлайн
    proj_dir = _rand_folder(workflow_dir, "Workflow")
    sub1 = f"{proj_dir.name}_tl1"
    sub2 = f"{proj_dir.name}_tl2"
    (comfy / sub1).mkdir(parents=True, exist_ok=True)
    (comfy / sub2).mkdir(parents=True, exist_ok=True)

    def _build_ref_chars(slots, sub, prefix, timeline):
        """Датасет-слоты → characters[] ПО ИНДЕКСУ СЛОТА (1-3 = @ref1-3),
        пустые слоты остаются пустыми (важно для маппинга @refN и порядка в ноде)."""
        chars = [
            {"images": [], "description": ""},
            {"images": [], "description": ""},
            {"images": [], "description": ""},
        ]
        for slot_idx, slot_paths in enumerate(slots[:3]):
            images = []
            srcs = []
            for rp in (slot_paths or [])[:2]:
                if rp and Path(rp).exists():
                    name = prefix + Path(rp).name
                    images.append({"name": name})
                    srcs.append(rp)
                    shutil.copy2(rp, comfy / sub / name)
                    shutil.copy2(rp, proj_dir / name)
            if images:
                desc_map = _caption_ref_images(srcs, timeline=timeline)
                desc_text = "\n".join(d for d in (desc_map.get(p) for p in srcs) if d)
                chars[slot_idx] = {"images": images, "description": desc_text}
        return chars

    ref_images1 = _build_ref_chars([_norm_files(ref0), _norm_files(ref1), _norm_files(ref2)], sub1, "ltxref_", timeline=1)
    ref_images2 = _build_ref_chars([_norm_files(ref2_0), _norm_files(ref2_1), _norm_files(ref2_2)], sub2, "tl2_ltxref_", timeline=2) if two else []

    def _copy_seg_images(paths, sub):
        seg_images = [None] * len(paths)
        slot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for i, p in enumerate(paths):
            if not p:
                continue
            src = Path(p)
            if not src.exists():
                continue
            suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
            name = f"{slot_labels[i]}_{suffix}_.png"
            shutil.copy2(src, comfy / sub / name)
            shutil.copy2(src, proj_dir / name)
            seg_images[i] = name
        return seg_images

    seg_images1 = _copy_seg_images([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9][:len(tl1_seqs)], sub1)
    seg_images2 = _copy_seg_images([t2_0, t2_1, t2_2, t2_3, t2_4, t2_5, t2_6, t2_7, t2_8, t2_9][:len(tl2_seqs)], sub2) if two else []

    timelines = [{
        "global_prompt":   global_prompt or "",
        "segment_prompts": tl1_seqs,
        "ref_images":      ref_images1,
        "seg_images":      seg_images1,
        "durations":       tl1_durs,
        "weights":         tl1_wts,
        "subfolder":       sub1,
        "frames":          tl1_frames,
    }]
    if two and tl2_seqs:
        timelines.append({
            "global_prompt":   global_prompt or "",
            "segment_prompts": tl2_seqs,
            "ref_images":      ref_images2,
            "seg_images":      seg_images2,
            "durations":       tl2_durs,
            "weights":         tl2_wts,
            "subfolder":       sub2,
            "frames":          tl2_frames,
        })

    try:
        out = tripple.create_sc_workflow(
            template_path   = workflow_dir / "SC.json",
            out_path        = proj_dir / "mod_SC.json",
            timelines       = timelines,
            fps             = 24,
            comfy_input_dir = comfy,
        )
    except Exception as e:
        return f"❌ Ошибка создания CS-JSON: {e}"

    (proj_dir / "video.txt").write_text((answer_text or "").strip(), encoding="utf-8")

    # ── рекомендации picture-промтов (Flux 2 Klein): сцены без картинок + анкоры ──
    pic_ok = False
    try:
        def _ref_descs(ref_images):
            descs = [(c.get("description") or "").strip() for c in ref_images]
            return (descs + ["", "", ""])[:3]

        tl1_refs = _ref_descs(ref_images1)
        tl2_refs = _ref_descs(ref_images2) if two else ["", "", ""]
        scenes = []
        _gi = 0
        for i, s in enumerate(tl1_seqs, 1):
            _gi += 1
            scenes.append((f"SCENE {_gi} (Timeline 1)", s,
                           bool(seg_images1[i - 1] if i - 1 < len(seg_images1) else None), tl1_refs))
        for i, s in enumerate(tl2_seqs, 1):
            _gi += 1
            scenes.append((f"SCENE {_gi} (Timeline 2)", s,
                           bool(seg_images2[i - 1] if i - 1 < len(seg_images2) else None), tl2_refs))
        pic_txt = _generate_picture_prompts(scenes)
        if pic_txt:
            (proj_dir / "picture_prompts.txt").write_text(pic_txt + "\n", encoding="utf-8")
            pic_ok = True
    except Exception as e:
        rag_core._debug_log(f"⚠️ picture_prompts.txt не создан: {e}")

    n_imgs1 = sum(1 for s in seg_images1 if s)
    n_imgs2 = sum(1 for s in seg_images2 if s)
    parts = [f"✅ CS-JSON: {out.name} → {proj_dir.name}",
             f"TL1: {len(tl1_seqs)} сцен, {tl1_frames} кадров, @ref {len(ref_images1)}, картинок {n_imgs1}"]
    if two:
        parts.append(f"TL2: {len(tl2_seqs)} сцен, {tl2_frames} кадров, @ref {len(ref_images2)}, картинок {n_imgs2}")
    if pic_ok:
        parts.append("picture_prompts.txt")
    return " | ".join(parts)


def export_ltx25_director_json(answer_text, n_seq, video_length, director_mode, prov,
                               ref0, ref1, ref2,
                               p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    """
    Собирает mod_ltx25_director.json из ответа LTX 2.5 Director:
    шаблон LTX_2_5_Director_CStest.json, нода LTXDirectorCS25, ОДИН таймлайн
    (принцип CS LTX Director, см. export_cs_json). JSON создаёт tripple.py.
    """
    if director_mode != "LTX 2.5 Director":
        return "⚠️ Сначала выбери сценарий «LTX 2.5 Director» и нажми «Сгенерировать»."

    global_prompt, seqs, durs, weights = tripple.parse_director_answer(answer_text or "")
    if not seqs:
        return ("⚠️ В ответе нет формата ---GLOBAL_PROMPT_START--- / ---SEQUENCE_N_START---. "
                "Сгенерируй промт в режиме LTX 2.5 Director.")

    n1 = max(int(n_seq) if n_seq else 5, 1)
    seqs = seqs[:n1]
    durs = durs[:n1]
    weights = weights[:n1]
    if not seqs:
        return "⚠️ Нет SEQUENCE — сгенерируй промт с количеством сцен ≥ 1."

    workflow_dir = Path(r"D:\claudeAgent\RAG\workflow")
    comfy = _comfy_input_dir()
    frames = tripple.frames_for_seconds(int(video_length) if video_length else 30)

    proj_dir = _rand_folder(workflow_dir, "LTX25Dir")
    sub = f"{proj_dir.name}_tl1"
    (comfy / sub).mkdir(parents=True, exist_ok=True)

    def _build_ref_chars(slots, prefix, timeline):
        chars = [{"images": [], "description": ""} for _ in range(3)]
        for slot_idx, slot_paths in enumerate(slots[:3]):
            images, srcs = [], []
            for rp in (slot_paths or [])[:2]:
                if rp and Path(rp).exists():
                    name = prefix + Path(rp).name
                    images.append({"name": name})
                    srcs.append(rp)
                    shutil.copy2(rp, comfy / sub / name)
                    shutil.copy2(rp, proj_dir / name)
            if images:
                dm = _caption_ref_images(srcs, timeline=timeline)
                dt = "\n".join(d for d in (dm.get(p) for p in srcs) if d)
                chars[slot_idx] = {"images": images, "description": dt}
        return chars

    def _copy_seg_images(paths):
        seg_images = [None] * len(paths)
        slot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for i, p in enumerate(paths):
            if not p:
                continue
            src = Path(p)
            if not src.exists():
                continue
            suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
            name = f"{slot_labels[i]}_{suffix}_.png"
            shutil.copy2(src, comfy / sub / name)
            shutil.copy2(src, proj_dir / name)
            seg_images[i] = name
        return seg_images

    ref_images = _build_ref_chars([_norm_files(ref0), _norm_files(ref1), _norm_files(ref2)], "ltxref_", timeline=1)
    seg_images = _copy_seg_images([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9][:len(seqs)])

    timelines = [{
        "global_prompt":   global_prompt or "",
        "segment_prompts": seqs,
        "ref_images":      ref_images,
        "seg_images":      seg_images,
        "durations":       durs,
        "weights":         weights,
        "subfolder":       sub,
        "frames":          frames,
    }]

    try:
        out = tripple.create_sc_workflow(
            template_path   = workflow_dir / "LTX_2_5_Director_CStest.json",
            out_path        = proj_dir / "mod_ltx25_director.json",
            timelines       = timelines,
            fps             = 24,
            comfy_input_dir = comfy,
            node_type       = "LTXDirectorCS25",
        )
    except Exception as e:
        return f"❌ Ошибка создания LTX 2.5 Director JSON: {e}"

    (proj_dir / "video.txt").write_text((answer_text or "").strip(), encoding="utf-8")

    n_imgs = sum(1 for s in seg_images if s)
    n_refs = sum(1 for c in ref_images if c["images"])
    return (f"✅ LTX 2.5 Director JSON: {out.name} → {proj_dir.name} | "
            f"{len(seqs)} сцен, {frames} кадров, @ref {n_refs}, картинок {n_imgs}")


def load_json_into_ui(json_path: str, n_sequences: int = 5):
    """
    Загружает ComfyUI-workflow-JSON обратно в UI (LTXDirector / PromptRelayEncodeTimeline):
      - картинки из сегментов (imageFile) раскидывает по img_slots
      - global_prompt + сегментные промты собирает в формат GLOBAL/SEQUENCE
    Кол-во видимых слотов = max(указанное n_sequences, сегментов в JSON), максимум 10:
      сегментов больше указанного → показываем все; меньше → добиваем пустыми видимыми слотами.
    Returns: (10 gr.update для img_slots, собранный промт, статус) — распакованные значения
    """
    # Убираем обрамляющие кавычки (одинарные/двойные), если пользователь вставил путь с ними
    json_path = (json_path or "").strip().strip("\"'")
    n_req = max(1, int(n_sequences) if n_sequences else 5)
    if not json_path:
        return *([gr.update(value=None, visible=True)]*n_req + [gr.update(value=None, visible=False)]*(10-n_req)), "", "⚠️ Укажи путь к JSON (например mod_high.json)."
    p = Path(json_path)
    if not p.exists():
        return *([gr.update(value=None, visible=True)]*n_req + [gr.update(value=None, visible=False)]*(10-n_req)), "", f"⚠️ Файл не найден: {p}"
    try:
        import json as _json
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return *([gr.update(value=None, visible=True)]*n_req + [gr.update(value=None, visible=False)]*(10-n_req)), "", f"❌ Не удалось прочитать JSON: {e}"

    # Находим ноду таймлайна: LTXDirector (whatdreamscost) или PromptRelayEncodeTimeline (kijai/ComfyUI-PromptRelay)
    director = None
    node_type = None
    for node in data.get("nodes", []):
        if node.get("type") in ("LTXDirector", "PromptRelayEncodeTimeline"):
            director = node
            node_type = node.get("type")
            break
    if director is None:
        return *([gr.update(value=None, visible=True)]*n_req + [gr.update(value=None, visible=False)]*(10-n_req)), "", "❌ Нода таймлайна не найдена в JSON (искал LTXDirector / PromptRelayEncodeTimeline)."

    widgets = director.get("widgets_values") or []
    timeline_data = {}
    global_prompt = ""
    if node_type == "PromptRelayEncodeTimeline":
        # kijai/ComfyUI-PromptRelay: global_prompt в widgets[0], таймлайн-JSON в widgets[2]
        if widgets and widgets[0]:
            global_prompt = str(widgets[0])
        if len(widgets) > 2 and isinstance(widgets[2], str):
            try:
                timeline_data = _json.loads(widgets[2])
            except Exception:
                timeline_data = {}
    else:
        # LTXDirector: таймлайн-JSON в widgets[6] (внутри может лежать global_prompt)
        if len(widgets) > 6 and isinstance(widgets[6], str):
            try:
                timeline_data = _json.loads(widgets[6])
            except Exception:
                timeline_data = {}
        # Версия 2.x (whatdreamscost 2.0.5) хранит таймлайн и в properties — фолбэк
        if not timeline_data:
            props = director.get("properties") or {}
            try:
                timeline_data = _json.loads(props.get("timeline_data") or "{}")
            except Exception:
                timeline_data = {}

    segments = timeline_data.get("segments", [])
    if not global_prompt:
        global_prompt = timeline_data.get("global_prompt", "")
    # Последний фолбэк: properties.global_prompt (LTXDirector 2.x)
    if not global_prompt:
        global_prompt = (director.get("properties") or {}).get("global_prompt", "")

    # Итоговое число видимых слотов: не меньше запрошенного, не больше 10
    n_show = max(n_req, len(segments))
    n_show = min(n_show, 10)

    # Собираем промт обратно в формат ---GLOBAL_PROMPT_START--- / ---SEQUENCE_N_START---
    lines = []
    if global_prompt:
        lines.append("---GLOBAL_PROMPT_START---")
        lines.append(global_prompt)
        lines.append("---GLOBAL_PROMPT_END---")
        lines.append("")

    # Куда искать картинки: рядом с JSON (папка проекта) или ComfyUiVid\\input
    search_dirs = [p.parent] + [Path(r"D:\ComfyUiVid\input")]
    img_updates = []
    for i in range(10):
        if i >= n_show:
            img_updates.append(gr.update(value=None, visible=False))
            continue
        seg = segments[i] if i < len(segments) else {}
        seg_prompt = seg.get("prompt", "")
        if seg_prompt:
            lines.append(f"---SEQUENCE_{i+1}_START---")
            lines.append(seg_prompt)
            lines.append(f"---SEQUENCE_{i+1}_END---")
            lines.append("")
        # Картинка сегмента
        img_file = seg.get("imageFile") or ""
        found = None
        if img_file:
            for d in search_dirs:
                cand = d / Path(img_file).name
                if cand.exists():
                    found = str(cand)
                    break
        # Пустой слот (нет сегмента или нет картинки) — видимый, можно загрузить
        img_updates.append(gr.update(value=found, visible=True))

    prompt_text = "\n".join(lines).strip()
    status = (f"✅ Загружено из {p.name}: {len(segments)} сегментов, "
              f"{sum(1 for u in img_updates if u.get('value'))} картинок. Видимых слотов: {n_show}.")
    return *img_updates, prompt_text, status


def update_image_slots(n_sequences: int, director_mode: str = "нет"):
    """
    Возвращает обновления видимости для image-слотов, timeline-колонок
    и чекбоксов в зависимости от выбранного количества сцен.
    В режиме TripleUpscale виден только 1 слот (стартовый кадр I2V).
    """
    n = int(n_sequences)
    if director_mode == "LTX 2.5 T2V":
        # чистый текст — слоты скрыты
        return ([gr.update(visible=False) for _ in range(10)]
                + [gr.update(visible=False) for _ in range(10)]
                + [gr.update(visible=False) for _ in range(200)])
    if director_mode == "TripleUpscale":
        n = 1
    elif director_mode == "LTX 2.5 First/Last":
        n = 2   # первый + последний кадр
    elif director_mode in ("Minimax H3", "Minimax H3 Anim"):
        n = 9   # 9 референс-картинок (<Picture 1..9>)
    if n == 1:
        pcts = ["0%"]
    else:
        step = 100 // (n - 1)
        pcts = [f"{i * step}%" for i in range(n)]
        pcts[-1] = "100%"

    if director_mode in ("Minimax H3", "Minimax H3 Anim"):
        slot_labels = [f"🖼️ Picture {i+1}" for i in range(10)]
    elif director_mode == "LTX 2.5 First/Last":
        slot_labels = (["🎞️ Первый кадр (START)", "🎞️ Последний кадр (END)"]
                       + [f"🎥 слот {i+1}" for i in range(2, 10)])
    else:
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
    for r in range(20):
        for c in range(10):
            cb_updates.append(gr.update(visible=(c < n)))

    return updates + col_labels + cb_updates


# ──────────────────────────────────────────────
# ИНТЕРФЕЙС
# ──────────────────────────────────────────────

GRID_CAPTION_CHOICES = ("с текстом (подписями)", "без текста (только кадр)")


def do_grid_split(image_path, vl_choice, captions_choice):
    """Split an uploaded grid and return a status message and gallery paths.

    Select Gemma only for ``vl_choice == "Gemma-4-E2B"``; otherwise use Qwen.
    A ``captions_choice`` string starting with "без" requests caption trimming.
    Save PNGs named with a number and three random letters under the splitter's
    default ``gridN`` output folder and record it in ``used.md``.
    Missing input or an exception from splitting returns an error status and
    an empty gallery; splitting errors include the last 12 collected messages.
    """
    if not image_path:
        return "❌ Bitte zuerst ein Gitter-Bild hochladen.", []
    logs = []
    try:
        folder, files = grid_splitter.split_image(
            image_path, vl=("gemma" if vl_choice == "Gemma-4-E2B" else "qwen"),
            with_captions=(captions_choice != GRID_CAPTION_CHOICES[1]), on_log=logs.append
        )
    except Exception as e:
        return "❌ " + str(e) + "\n\n" + "\n".join(logs[-12:]), []
    msg = f"✅ {len(files)} Bilder → {folder}  (in used.md eingetragen)"
    return msg, [str(f) for f in files]


def build_ui():
    """Build and return the Gradio Blocks interface with its event handlers connected."""
    _STEAM_THEME = gr.themes.Base(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=["Georgia", "Times New Roman", "serif"],
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="linear-gradient(160deg, #241a10 0%, #1b130b 55%, #120c07 100%)",
        background_fill_primary="#2b1f13",
        background_fill_secondary="#37281a",
        block_background_fill="linear-gradient(180deg, rgba(64,46,28,0.92), rgba(43,31,19,0.94))",
        block_border="2px solid #8a6a3b",
        block_shadow="0 6px 24px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,214,150,0.15)",
        block_title_text_color="#e8c98a",
        block_label_text_color="#d9b878",
        body_text_color="#eadbc0",
        input_background_fill="#221810",
        input_border="1px solid #7a5c33",
        button_primary_background_fill="linear-gradient(180deg, #c98a3d, #8f5a22)",
        button_primary_border="1px solid #e0aa5a",
        button_secondary_background_fill="#3b2c1c",
        button_secondary_border="1px solid #8a6a3b",
    )

    _LAYOUT_CSS = """
/* ══ Steampunk-Anstrich: Messing, Leder, Nieten ══ */
.gradio-container { letter-spacing: 0.2px; }
h1, h2, h3 { color: #f0cf8f !important; text-shadow: 0 2px 6px rgba(0,0,0,0.6); }
h1 { border-bottom: 2px solid #8a6a3b; padding-bottom: 10px; letter-spacing: 1px; }
.block, .form > .block {
    border-radius: 10px !important;
    position: relative;
}
/* Nietenecken */
.panel::before, .gap > .block::after {
    content: "";
    position: absolute; top: 6px; right: 8px;
    width: 7px; height: 7px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #ffd98a, #7a5c33 70%, #3a2a16);
    box-shadow: -18px 0 0 -0.5px #6b4f2a, 0 0 4px rgba(0,0,0,0.6);
    pointer-events: none; z-index: 5;
}
button { text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }
textarea, input[type="text"] { font-family: Georgia, 'Times New Roman', serif; }
footer { display: none; }
/* Блок генерации: запрос на всю ширину, ниже — сетка из 4 колонок, центрированно */
#gen_block { max-width: 1320px; margin: 0 auto !important; }
#ctrl_grid > .form,
#ctrl_grid:not(:has(> .form)) {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    align-items: start;
}
#ctrl_grid > .form > .block { min-width: 0 !important; }
#ctrl_grid > .form > .block:has(input[type="range"]) { grid-column: span 2; }
@media (max-width: 1000px) {
    #ctrl_grid > .form, #ctrl_grid:not(:has(> .form)) { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 600px) {
    #ctrl_grid > .form, #ctrl_grid:not(:has(> .form)) { grid-template-columns: minmax(0, 1fr); }
}
"""

    with gr.Blocks(title="RAG — Генератор сценариев", theme=_STEAM_THEME, css=_LAYOUT_CSS) as demo:

        gr.Markdown("# ⚙️🎬 RAG — Генератор сценариев")

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
                value       = rag_core.load_settings().get("MODEL_PATH") or None,
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
        with gr.Column(elem_id="gen_block"):
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label       = "Вопрос / тема / запрос",
                    placeholder = "Напиши рассказ про усталость от работы на 200 слов",
                    lines       = 3,
                )
            with gr.Column(scale=1, elem_id="ctrl_grid"):
                style_dd = gr.Dropdown(
                    label   = "Стиль",
                    info    = "Влияет только в Director/CS/LTX 2.5 Director и без сценария",
                    choices = ["реализм", "мрачный", "юмор", "поток сознания"],
                    value   = "реализм",
                )
                mode_dd = gr.Dropdown(
                    label   = "Режим",
                    info    = "Только если «Формат LTX 2.3» = нет",
                    choices = ["художественный", "аналитический"],
                    value   = "художественный",
                )
                # RAG-коллекция: "выкл" = не грузить эмбеддинг-модель вообще
                try:
                    _cols = rag_core.list_collections()
                except Exception:
                    _cols = []
                rag_collection_dd = gr.Dropdown(
                    label   = "🗄️ RAG коллекция (выкл = без RAG)",
                    choices = ["выкл (без RAG)"] + _cols,
                    value   = "выкл (без RAG)",
                )
                lang_dd = gr.Dropdown(
                    label   = "Язык вывода",
                    choices = ["русский", "немецкий", "английский", "испанский", "польский", "нидерландский", "французский", "китайский", "латинский"],
                    value   = "русский",
                )
                env = rag_core.load_env()
                cloud_choices = ["auto"]
                first_available = "auto"
                for pn, cfg in rag_core.CLOUD_PROVIDERS.items():
                    if env.get(cfg["key_var"], "").strip():
                        cloud_choices.append(pn)
                        if first_available == "auto":
                            first_available = pn
                cloud_dd = gr.Dropdown(
                    label   = "Облачный API",
                    choices = cloud_choices,
                    value   = "auto",   # immer 'auto': lokaler Server hat Vorrang, Cloud nur als Fallback
                )
                script_dd = gr.Dropdown(
                    label   = "Формат LTX 2.3",
                    choices = ["нет", "сценарий", "создать песню"],
                    value   = "сценарий",
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
                    info    = "Не влияет на First/Last, T2V и Minimax",
                    choices = base_narrators + music_narrators,
                    value   = "короткий фильм",
                    visible = True,
                )
                director_dd = gr.Dropdown(
                    label   = "Director Mode",
                    choices = ["нет", "Director Mode", "TripleUpscale", "CS LTX Director", "LTX 2.5 Director", "LTX 2.5 First/Last", "LTX 2.5 T2V", "Minimax H3", "Minimax H3 Anim"],
                    value   = "Director Mode",
                    visible = True,
                )
                cs_timelines_dd = gr.Dropdown(
                    label   = "🎞️ Таймлайнов CS (LTXDirectorCS)",
                    choices = [1, 2],
                    value   = 1,
                    visible = False,
                )
                n_seq_dd = gr.Dropdown(
                    label   = "🔢 Количество сцен (Sequences)",
                    choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    value   = 5,
                )
                n_seq_tl2_dd = gr.Dropdown(
                    label   = "🔢 Сцен во 2-м таймлайне (TL2)",
                    choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    value   = 3,
                    visible = False,
                )
                singer_dd = gr.Dropdown(
                    label   = "🎙️ Sänger / Вокалист",
                    choices = ["женщина", "мужчина", "дуэт"],
                    value   = "женщина",
                    visible = False,
                )
                verses_dd = gr.Dropdown(
                    label   = "📜 Куплеты (Strophen)",
                    choices = [1, 2, 3, 4, 5, 6],
                    value   = 2,
                    visible = False,
                )
                song_style_dd = gr.Dropdown(
                    label   = "🎤 Musikstil (20 Styles)",
                    choices = [label for label, _ in _song_styles] or ["Keine Styles gefunden"],
                    value   = (_song_styles[0][0] if _song_styles else None),
                    visible = False,
                )
                video_len_sl = gr.Slider(
                    label   = "⏱️ Videolänge (Sekunden)",
                    minimum = 1,
                    maximum = 60,
                    step    = 1,
                    value   = 30,
                )
                word_sl = gr.Slider(
                    label   = "Длина (слов)",
                    minimum = 50,
                    maximum = 1200,
                    step    = 25,
                    value   = 800,
                )

        # ── БЛОК: Timeline матрица приёмов ────────────────────────────
        with gr.Column(visible=True) as timeline_column:
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
                    "👀 Eye Level (уровень глаз — нейтральный реализм)",
                    "🔻 High Angle (сверху вниз — уязвимость/малость)",
                    "🦅 Bird's Eye View (вид с высоты — масштаб/панорама)",
                    "🐜 Worm's Eye View (снизу — монументальность/угроза)",
                    "🤝 Over-the-Shoulder (через плечо — связь персонажей)",
                    "🏞️ Establishing Shot (устанавливающий — контекст места)",
                    "🌅 Wide Shot (широкий план — окружение)",
                    "🎯 Close-Up (крупный план — эмоции)",
                    "👥 Two Shot (двое в кадре — взаимодействие)",
                    "😲 Reaction Shot (реакция — ответный взгляд)",
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
                for r in range(20):
                    with gr.Row():
                        gr.Markdown(f"**{technique_labels[r]}**")
                        row_cbs = []
                        for c in range(10):
                            cb = gr.Checkbox(label="Ctrl", value=False, interactive=True, container=False, min_width=40)
                            row_cbs.append(cb)
                        cbs.append(row_cbs)

        # Каскадная логика: только 1 чекбокс на колонку
        for r in range(20):
            for c in range(10):
                other_outputs = [cbs[rr][c] for rr in range(20) if rr != r]
                def _make_cascade(row=r, col=c):
                    def _handler(checked):
                        updates = []
                        for rr in range(20):
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

                # ── CS LTX Director: @ref1/@ref2/@ref3 — датасет (до 2 картинок на слот) ──
                with gr.Row(visible=False) as cs_ref_row:
                    ref_slots = [
                        gr.File(
                            label      = f"@ref{i+1} (SC Director — датасет: до 2 картинок)",
                            file_count = "multiple",
                            file_types = [".png", ".jpg", ".jpeg", ".webp"],
                            visible    = True,
                        )
                        for i in range(3)
                    ]

                # ── CS LTX Director, Timeline 2: @ref1-3 — датасет (до 2 картинок на слот) ──
                with gr.Row(visible=False) as cs_ref_row2:
                    ref_slots2 = [
                        gr.File(
                            label      = f"TL2 @ref{i+1} (Timeline 2 — датасет: до 2)",
                            file_count = "multiple",
                            file_types = [".png", ".jpg", ".jpeg", ".webp"],
                            visible    = True,
                        )
                        for i in range(3)
                    ]

                with gr.Row(visible=False) as cs_img_row2:
                    img_slots_tl2 = [
                        gr.Image(
                            label   = f"🎥 TL2 {['0% (Старт)', '11%', '22%', '33%', '44%', '56%', '67%', '78%', '89%', '100% (Финал)'][i]}",
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

        # ── БЛОК: Grid-Splitter ──────────────────────────────────────
        gr.Markdown("---")
        with gr.Accordion("✂️ Grid-Splitter (Storyboard-Gitter → Einzelbilder)", open=False):
            gr.Markdown(
                "Schneidet ein Gitter-Bild per **Qwen3-VL** sauber in Einzelbilder (mit Beschriftung). "
                "Dateinamen: Nummer + 3 Zufallsbuchstaben, Ordner `workflow\\gridN` (erster freier Name laut `used.md`). "
                "<small>Ohne laufenden Vision-Server startet das Tool einen eigenen und beendet ihn danach "
                "(bei wenig freiem VRAM langsamer auf CPU).</small>"
            )
            with gr.Row():
                grid_img = gr.Image(label="🖼️ Gitter-Bild", type="filepath", height=320, scale=2)
                with gr.Column(scale=1):
                    grid_model_dd = gr.Dropdown(
                        label="Vision-Modell", choices=["Qwen3-VL-4B", "Gemma-4-E2B"],
                        value="Qwen3-VL-4B",
                    )
                    grid_captions_dd = gr.Dropdown(
                        label="✂️ Нарезать с текстом или без (подписи под кадрами)",
                        choices=GRID_CAPTION_CHOICES,
                        value=GRID_CAPTION_CHOICES[0],
                    )
                    grid_btn = gr.Button("✂️ Gitter schneiden", variant="primary")
                    grid_status = gr.Textbox(label="Status", interactive=False, lines=3)
            grid_gallery = gr.Gallery(label="Ergebnis", columns=4, height=420, object_fit="contain")
            grid_btn.click(
                fn      = do_grid_split,
                inputs  = [grid_img, grid_model_dd, grid_captions_dd],
                outputs = [grid_status, grid_gallery],
            )

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
        with gr.Row():
            export_tripple_btn = gr.Button("📤 Export Tripple.json (TripleUpscale)", variant="secondary", scale=1)
            export_cs_btn      = gr.Button("📤 Export CS-json (SC Director)", variant="secondary", scale=1)
            export_ltx25_dir_btn = gr.Button("📤 Export LTX 2.5 Director (CS25)", variant="secondary", scale=1)
            export_ltx25_btn   = gr.Button("📤 Export LTX 2.5 (First/Last)", variant="secondary", scale=1)
            export_t2v_btn     = gr.Button("📤 Export LTX 2.5 T2V", variant="secondary", scale=1)
            export_minimax_btn = gr.Button("📤 Export MiniMax H3 (Reference)", variant="secondary", scale=1)
        with gr.Row():
            json_path_tb = gr.Textbox(
                label       = "📥 Путь к JSON (загрузка обратно в UI)",
                placeholder = r"D:\claudeAgent\RAG\workflow\<projektordner>\mod_high.json",
                scale       = 4,
            )
            load_json_btn = gr.Button("📥 Load JSON → UI", variant="primary", scale=1)
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
                       narrator_dd, director_dd, n_seq_dd, video_len_sl, song_style_dd, singer_dd, verses_dd, cloud_dd, sys_prompt_box] + img_slots + ref_slots + all_cbs + [rag_collection_dd, cs_timelines_dd, n_seq_tl2_dd] + ref_slots2 + img_slots_tl2,
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
                gr.update(visible=(v == "создать песню")),
                gr.update(visible=(v == "создать песню")),
                gr.update(visible=(v == "создать песню")),
            ),
            inputs  = [script_dd],
            outputs = [narrator_dd, director_dd, timeline_column, singer_dd, verses_dd, song_style_dd],
        )

        topics_btn.click(
            fn      = do_topics,
            inputs  = [question, cloud_dd, rag_collection_dd],
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
            inputs  = [n_seq_dd, director_dd],
            outputs = img_slots + tl_col_refs + all_cbs,
        )

        # ── СОБЫТИЕ: переключение сценария (нет / Director / TripleUpscale / CS LTX Director) ──
        director_dd.change(
            fn      = lambda v, n: (update_image_slots(n, v)
                                    + [gr.update(visible=(v in ("CS LTX Director", "LTX 2.5 Director"))),
                                       gr.update(visible=(v == "CS LTX Director"), value=1),
                                       gr.update(visible=False), gr.update(visible=False),
                                       gr.update(visible=False)]
                                    + [gr.update(visible=False)] * 10),
            inputs  = [director_dd, n_seq_dd],
            outputs = img_slots + tl_col_refs + all_cbs + [cs_ref_row, cs_timelines_dd, n_seq_tl2_dd, cs_ref_row2, cs_img_row2] + img_slots_tl2,
        )

        # ── СОБЫТИЕ: 1 или 2 таймлайна CS (LTXDirectorCS) ──
        cs_timelines_dd.change(
            fn      = lambda v: (
                gr.update(visible=(v == 2)),
                gr.update(visible=(v == 2)),
                gr.update(visible=(v == 2)),
                gr.update(label="🔢 Сцен в 1-м таймлайне (TL1)" if v == 2 else "🔢 Количество сцен (Sequences)"),
                gr.update(label="🔢 Сцен во 2-м таймлайне (TL2)"),
            ),
            inputs  = [cs_timelines_dd],
            outputs = [n_seq_tl2_dd, cs_ref_row2, cs_img_row2, n_seq_dd, n_seq_tl2_dd],
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
            inputs  = img_slots + [answer_box],
            outputs = [answer_export_status],
        )

        json_btn.click(
            fn      = do_json_workflow,
            inputs  = [],
            outputs = [answer_export_status],
        )

        # ── Export Tripple.json (TripleUpscale) / Export CS-json (SC Director) ──
        export_tripple_btn.click(
            fn      = export_tripple_json,
            inputs  = [answer_box, img_slots[0], video_len_sl, director_dd],
            outputs = [answer_export_status],
        )

        export_cs_btn.click(
            fn      = export_cs_json,
            inputs  = [answer_box, n_seq_dd, n_seq_tl2_dd, video_len_sl, director_dd, cloud_dd, cs_timelines_dd] + ref_slots + ref_slots2 + img_slots + img_slots_tl2,
            outputs = [answer_export_status],
        )

        export_ltx25_dir_btn.click(
            fn      = export_ltx25_director_json,
            inputs  = [answer_box, n_seq_dd, video_len_sl, director_dd, cloud_dd] + ref_slots + img_slots,
            outputs = [answer_export_status],
        )

        export_ltx25_btn.click(
            fn      = export_ltx25_json,
            inputs  = [answer_box, img_slots[0], img_slots[1], video_len_sl, director_dd],
            outputs = [answer_export_status],
        )

        export_t2v_btn.click(
            fn      = export_t2v_json,
            inputs  = [answer_box, video_len_sl, director_dd],
            outputs = [answer_export_status],
        )

        export_minimax_btn.click(
            fn      = export_minimax_json,
            inputs  = [answer_box, video_len_sl, director_dd] + img_slots[:9],
            outputs = [answer_export_status],
        )

        # ── Load JSON → UI (картинки + промт обратно в слоты) ─────────
        load_json_btn.click(
            fn      = load_json_into_ui,
            inputs  = [json_path_tb, n_seq_dd],
            outputs = img_slots + [answer_box, answer_export_status],
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
    rag_core._debug_log("WebUI gestartet — warte auf Generierung...")
    print("[RAG] Запускаю WebUI...")
    settings = rag_core.load_settings()
    model_path = (settings.get("MODEL_PATH") or "").strip()
    if model_path:
        print(f"[RAG] Settings: MODEL_PATH = {model_path}")
        if not rag_core.llama_alive() and not rag_core.kobold_alive():
            print("[RAG] Автоматический запуск llama-server mit Modell aus settings.txt...")
            rag_core.start_llama_server(model_path)
    else:
        print("[RAG] Settings: MODEL_PATH leer — Modell manuell in der WebUI wählen.")
    print(f"[RAG] Бэкенд: {backend_status()}")
    ui = build_ui()
    ui.launch(
        server_name = rag_core.NET.webui_host,   # WEBUI_HOST: 0.0.0.0 = Zugriff im LAN, Standard
        server_port = rag_core.NET.webui_port,   # WEBUI_PORT-Umgebung, Standard 7960
        inbrowser   = True,
    )
