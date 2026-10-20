#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tripple.py — извлечение промтов/длительности И создание ComfyUI-JSON.

Режимы (CLI):
    python tripple.py                          # extract: читает Tripple.json рядом со скриптом
    python tripple.py extract [путь\к.json]    # то же самое явно
    python tripple.py create ...               # создаёт Tripple-JSON (TripleUpscale) из шаблона Tripple.json
    python tripple.py create-sc ...            # создаёт CS-JSON (LTXDirectorCS) из шаблона SC.json
    python tripple.py create-minimax-ref ...   # MiniMax H3 Reference-to-Video из Minimax_reference_pictures.json

create — принцип update_workflow.py: берёт шаблон workflow (Tripple.json),
вставляет промты (4 поля ноды Tripple + FLUX_START_IMAGE → flux_start_image.txt),
fps/кадры и стартовое изображение (I2V). Экспорт создаёт новую папку
triple_<3 случайных символа>.

create-sc — принцип high.json: берёт шаблон SC.json и по списку timelines
(1 или 2 таймлайна, каждый со своим global_prompt/сегментами/@ref-картинками
с VL-описаниями/картинками сегментов/длительностью) обновляет ноды
LTXDirectorCS слева направо (timelines[0] → самая левая нода и т.д.),
синхронизирует widgets_values и properties, сохраняет mod_SC.json.
При 1 таймлайне трогается только первая (левая) нода.

Результат extract печатается в консоль и сохраняется в файл
<имя_json>_extracted.txt рядом с JSON.
"""

import json
import re
import sys
import time
import random
import shutil
from pathlib import Path

# Тип кастомной ноды «Tripple» — держит 4 промта + FPS + кол-во кадров
TRIPPLE_NODE_TYPE = "b0b6cb1c-d8aa-47bb-93fc-a5bf22978947"

# Индексы в widgets_values ноды Tripple
W_GLOBAL_POSITIVE = 0
W_SCENE_PROMPT    = 1
W_GLOBAL_NEGATIVE = 2
W_SCENE_NEGATIVE  = 3
W_FPS             = 4
W_FRAMES          = 5

# Индексы в widgets_values ноды LTXDirectorCS (SC.json)
CS_START_SEC  = 0
CS_END_SEC    = 1
CS_DUR_SEC    = 2
CS_START_FR   = 3
CS_END_FR     = 4
CS_DUR_FR     = 5
CS_TIMELINE   = 6
CS_LOCAL_PROMPTS = 7
CS_SEG_LENGTHS   = 8
CS_FPS        = 14

FIELD_META = [
    ("global_positive", "🟢 GLOBAL POSITIVE"),
    ("scene_prompt",    "🔵 SCENE PROMPT"),
    ("global_negative", "🔴 GLOBAL NEGATIVE"),
    ("scene_negative",  "🟠 SCENE NEGATIVE"),
]

DEFAULT_JSON = Path(__file__).resolve().parent / "Tripple.json"
DEFAULT_SC   = Path(__file__).resolve().parent / "SC.json"
DEFAULT_LTX25 = Path(__file__).resolve().parent / "LTX 2.5 First and Last Frame Video Generation .json"
DEFAULT_T2V = Path(__file__).resolve().parent / "Text-to-Video Fast LTX2.5 Sync Audio & Video.json"
COMFY_INPUT_DIR = Path(r"D:\ComfyUiVid\input")
DEFAULT_MINIMAX_REF = Path(__file__).resolve().parent / "Minimax_reference_pictures.json"


# ══════════════════════════════════════════════════
# ОБЩЕЕ
# ══════════════════════════════════════════════════

def iter_nodes(workflow):
    """Все ноды: верхнеуровневые + внутри групп + внутри subgraph-определений (definitions)."""
    for node in workflow.get("nodes", []):
        yield node
    for group in workflow.get("groups", []):
        for node in group.get("nodes", []):
            yield node
    for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []) or []:
        for node in subgraph.get("nodes", []):
            yield node


def as_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def find_nodes(workflow, node_type):
    return [n for n in iter_nodes(workflow) if n.get("type") == node_type]


def frames_for_seconds(seconds, fps=24):
    """LTX-совместимая длина в кадрах: ближайшее к seconds*fps число вида 48n+1."""
    seconds = max(1, int(seconds or 1))
    target = seconds * fps
    n = max(1, round((target - 1) / 48))
    return 48 * n + 1


def compute_segment_lengths(total_frames, n, durations=None, weights=None):
    """Длины сегментов (в кадрах), сумма = total_frames.

    Приоритет: 1) weights (DURATION_WEIGHT 1-5 от LLM, честный пейсинг) —
    пропорционально весу, с нижним порогом 1 кадр; 2) durations (секунды);
    3) поровну на всех, остаток в последнем сегменте.
    """
    total_frames = int(total_frames or 0)
    if n <= 0:
        return []
    # 1) DURATION_WEIGHT 1-5 — пропорционально весу
    if weights and len(weights) == n and all(isinstance(w, int) and 1 <= w <= 5 for w in weights):
        total_w = sum(weights)
        lengths = [max(1, round(total_frames * w / total_w)) for w in weights]
        diff = total_frames - sum(lengths)
        if diff != 0:
            lengths[-1] = max(1, lengths[-1] + diff)
        return lengths
    # 2) DURATION в секундах
    if durations and len(durations) == n and all(d and d > 0 for d in durations):
        total_sec = sum(durations)
        lengths = [max(1, round(total_frames * d / total_sec)) for d in durations]
        diff = total_frames - sum(lengths)
        if diff != 0:
            lengths[-1] = max(1, lengths[-1] + diff)
        return lengths
    # 3) поровну
    seg_len = max(1, total_frames // n)
    lengths = [seg_len] * n
    if total_frames:
        lengths[-1] = total_frames - seg_len * (n - 1)
    return lengths


# ══════════════════════════════════════════════════
# EXTRACT (промты + длительность из готового JSON)
# ══════════════════════════════════════════════════

def find_tripple_nodes(workflow):
    return find_nodes(workflow, TRIPPLE_NODE_TYPE)


def extract_prompts(node):
    """4 поля промта из widgets_values ноды Tripple."""
    wv = node.get("widgets_values") or []

    def val(i):
        return as_text(wv[i] if i < len(wv) else None)

    return {
        "global_positive": val(W_GLOBAL_POSITIVE),
        "scene_prompt":    val(W_SCENE_PROMPT),
        "global_negative": val(W_GLOBAL_NEGATIVE),
        "scene_negative":  val(W_SCENE_NEGATIVE),
    }


def extract_duration(node):
    """(fps, frames) из ноды Tripple; None, если значений нет."""
    wv = node.get("widgets_values") or []

    def num(i):
        try:
            return float(wv[i])
        except (TypeError, IndexError, ValueError):
            return None

    return num(W_FPS), num(W_FRAMES)


def fallback_duration(workflow):
    """Запасной источник длительности, если ноды Tripple нет:
    PrimitiveFloat 'fps' / PrimitiveInt 'number of frames' /
    EmptyLTXVLatentVideo (length) / LTXVEmptyLatentAudio (fps)."""
    fps = frames = None
    fps_src = frames_src = None
    for node in iter_nodes(workflow):
        ntype = node.get("type") or ""
        title = (node.get("title") or "").lower()
        wv = node.get("widgets_values") or []
        if ntype == "PrimitiveFloat" and "fps" in title and fps is None:
            try:
                fps = float(wv[0])
                fps_src = node.get("id")
            except (TypeError, IndexError, ValueError):
                pass
        elif ntype == "PrimitiveInt" and "frame" in title and frames is None:
            try:
                frames = float(wv[0])
                frames_src = node.get("id")
            except (TypeError, IndexError, ValueError):
                pass
        elif ntype == "EmptyLTXVLatentVideo" and frames is None:
            try:
                frames = float(wv[2])
                frames_src = node.get("id")
            except (TypeError, IndexError, ValueError):
                pass
        elif ntype == "LTXVEmptyLatentAudio" and fps is None:
            try:
                fps = float(wv[1])
                fps_src = node.get("id")
            except (TypeError, IndexError, ValueError):
                pass
    return fps, frames, fps_src, frames_src


def split_scene_segments(scene_prompt):
    """Разбивает сценарный промт на таймлайн-сегменты [mm:ss–mm:ss]."""
    if not scene_prompt.strip():
        return []
    pattern = re.compile(
        r"\[(\d{1,2}:\d{2})\s*(?:–|—|-|:)\s*(\d{1,2}:\d{2})\]\s*(.*?)(?=\n\s*\[|\Z)",
        re.DOTALL,
    )
    segments = []
    for m in pattern.finditer(scene_prompt):
        text = re.sub(r"\s+", " ", m.group(3)).strip()
        segments.append({"time": f"{m.group(1)}–{m.group(2)}", "text": text})
    return segments


def fallback_prompts(workflow):
    """Если ноды Tripple нет — классифицируем CLIPTextEncode-ноды
    по заголовкам и лейблам входов (GLOBAL POSITIVE / SCENE PROMPT / ...)."""
    fields = {name: [] for name, _ in FIELD_META}
    unknown = []
    for node in iter_nodes(workflow):
        if node.get("type") != "CLIPTextEncode":
            continue
        wv = node.get("widgets_values") or []
        text = as_text(wv[0] if wv else None)
        marker = " ".join(
            filter(None, [node.get("title"), *[i.get("label") for i in node.get("inputs", [])]])
        )
        upper = marker.upper()
        if "GLOBAL POSITIVE" in upper:
            fields["global_positive"].append(text)
        elif "GLOBAL NEGATIVE" in upper:
            fields["global_negative"].append(text)
        elif "SCENE PROMPT" in upper or "SZENEN-PROMPT" in upper or "NORMALER SZENEN" in upper:
            fields["scene_prompt"].append(text)
        elif "SCENE NEGATIVE" in upper or "SZENEN-NEGATIV" in upper:
            fields["scene_negative"].append(text)
        else:
            unknown.append((node.get("id"), node.get("title"), text))
    return fields, unknown


def format_duration(fps, frames, source):
    if fps is None or frames is None or fps == 0:
        return "🎬 Длительность видео: не найдена"
    dur = frames / fps
    dur_str = str(int(dur)) if abs(dur - round(dur)) < 1e-9 else f"{dur:.2f}"
    f = int(fps) if fps == int(fps) else fps
    fr = int(frames) if frames == int(frames) else frames
    return f"🎬 Длительность видео: {dur_str} сек ({fr} кадров @ {f} fps) — {source}"


def extract_main(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 60)
    emit(f"🎬 {Path(json_path).name} — промты и длительность видео")
    emit("=" * 60)

    triple_nodes = find_tripple_nodes(workflow)
    found_any = False

    if triple_nodes:
        for node in triple_nodes:
            nid = node.get("id")
            emit(f"\n📦 Нода «Tripple» (id: {nid})")
            prompts = extract_prompts(node)
            for name, icon in FIELD_META:
                emit(f"\n{icon}:")
                text = prompts[name]
                emit(text if text.strip() else "(пусто)")
                if name == "scene_prompt" and text.strip():
                    segments = split_scene_segments(text)
                    if segments:
                        emit("\n   Сегменты сцены:")
                        for i, seg in enumerate(segments, 1):
                            emit(f"   {i}. [{seg['time']}] {seg['text']}")
            fps, frames = extract_duration(node)
            emit("\n" + format_duration(fps, frames, f"нода Tripple (id {nid})"))
            found_any = True
    else:
        fields, unknown = fallback_prompts(workflow)
        if any(fields.values()):
            emit("\n📦 CLIPTextEncode-ноды (нода Tripple не найдена):")
            for name, icon in FIELD_META:
                vals = fields[name]
                if vals:
                    emit(f"\n{icon}:")
                    emit("\n---\n".join(vals))
            for nid, title, text in unknown:
                emit(f"\n❔ CLIPTextEncode (id: {nid}, {title or 'без названия'}):")
                emit(text if text.strip() else "(пусто)")
            found_any = True
        fps, frames, fps_src, frames_src = fallback_duration(workflow)
        if fps is not None or frames is not None:
            parts = []
            if fps_src is not None:
                parts.append(f"fps ({fps_src})")
            if frames_src is not None:
                parts.append(f"кадры ({frames_src})")
            src = " + ".join(parts) or "fallback"
            emit("\n" + format_duration(fps, frames, src))
            found_any = True

    if not found_any:
        emit("\n⚠️ Ни промты, ни длительность в этом JSON не найдены.")

    out_path = Path(json_path).with_name(Path(json_path).stem + "_extracted.txt")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n💾 Сохранено: {out_path}")


# ══════════════════════════════════════════════════
# PARSE ответов LLM
# ══════════════════════════════════════════════════

# Толерантный разделитель: ---GLOBAL_PROMPT_START--- / GLOBAL_PROMPT_START /
# GLOBAL_PROMPT\_START (экранированное подчёркивание) — как в update_workflow.py
_TOL = r"[\\\s_]*"


def _field_re(field: str):
    """(start_re, end_re) для поля вида ---GLOBAL_POSITIVE_START--- / ---..._END---."""
    return (
        rf"(?:-{{3}}\s*)?{field}{_TOL}START\s*(?:-{{3}})?",
        rf"(?:-{{3}}\s*)?{field}{_TOL}END\s*(?:-{{3}})?",
    )


def parse_tripple_answer(text):
    """
    Разбирает ответ LLM в формате TripleUpscale:
        ---GLOBAL_POSITIVE_START--- ... ---GLOBAL_POSITIVE_END---
        ---SCENE_PROMPT_START--- ... ---SCENE_PROMPT_END---
        ---GLOBAL_NEGATIVE_START--- ... ---GLOBAL_NEGATIVE_END---
        ---SCENE_NEGATIVE_START--- ... ---SCENE_NEGATIVE_END---   (опционально)
    Возвращает dict {global_positive, scene_prompt, global_negative, scene_negative}
    (значения могут быть None, если маркеры отсутствуют).
    """
    fields = {
        "global_positive": "GLOBAL[\\\s_]*POSITIVE",
        "scene_prompt":    "SCENE[\\\s_]*PROMPT",
        "global_negative": "GLOBAL[\\\s_]*NEGATIVE",
        "scene_negative":  "SCENE[\\\s_]*NEGATIVE",
        "flux_start_image": "FLUX[\\\s_]*START[\\\s_]*IMAGE",
    }
    result = {}
    for key, field in fields.items():
        start, end = _field_re(field)
        m = re.search(rf"{start}\s*(.*?)\s*{end}", text, re.DOTALL | re.IGNORECASE)
        result[key] = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    return result


_DURATION_TAG_RE = re.compile(r"^\s*\[\s*DURATION\s*:\s*([\d.]+)\s*s?\s*\]\s*", re.IGNORECASE)
_WEIGHT_TAG_RE = re.compile(r"^\s*DURATION\s*_?\s*WEIGHT\s*:\s*([1-5])\s*$", re.IGNORECASE)


def parse_director_answer(text):
    """
    Разбирает ответ LLM в формате Director / CS LTX Director:
        ---GLOBAL_PROMPT_START--- ... ---GLOBAL_PROMPT_END---
        ---SEQUENCE_N_START--- [DURATION: X.Xs] ... ---SEQUENCE_N_END---
    Тег [DURATION: ...] — разметка длинительности от LLM (опциональный), вырезается
    из текста сегмента и возвращается отдельно.

    ФОЛБЭК: маленькие локальные модели иногда теряют маркеры ---SEQUENCE_N_START---
    (особенно после второго/третьего прохода-ревью), но почти всегда честно выводят
    ---SEQUENCE_N_END---. Если строгая пара START/END нашла меньше сегментов, чем есть
    маркеров END, применяется фолбэк: всё между концом GLOBAL_PROMPT_END (или началом
    текста) и каждым очередным END — один сегмент; хвост после последнего END (если
    за ним не следует ещё один END) — конец таймлайна, отбрасывается.

    Возвращает (global_prompt, [prompts], [durations]) — durations[i] есть float секунд
    или None, если тег для этого сегмента отсутствует.
    """
    global_prompt = None
    gp_start, gp_end = _field_re(r"GLOBAL[\\\s_]*PROMPT")
    gm = re.search(rf"{gp_start}\s*(.*?)\s*{gp_end}", text, re.DOTALL | re.IGNORECASE)
    if gm:
        global_prompt = re.sub(r"\s+", " ", gm.group(1)).strip()

    seq_start = r"(?:-{3}\s*)?SEQUEN[\\\s_]*[CT][\\\s_]*[A-Z]*[\\\s_]*\d+[\\\s_]*START\s*(?:-{3})?"
    seq_end   = r"(?:-{3}\s*)?SEQUEN[\\\s_]*[CT][\\\s_]*[A-Z]*[\\\s_]*\d+[\\\s_]*END\s*(?:-{3})?"

    def _extract(block):
        block = block.strip()
        dur = None
        dm = _DURATION_TAG_RE.match(block)
        if dm:
            try:
                dur = float(dm.group(1))
            except ValueError:
                dur = None
            block = block[dm.end():].strip()
        # DURATION_WEIGHT: N (1-5) — строка в конце блока, вырезается из промпта
        weight = None
        keep = []
        for ln in block.splitlines():
            wm = _WEIGHT_TAG_RE.match(ln.strip())
            if wm:
                weight = int(wm.group(1))
                continue
            keep.append(ln)
        block = "\n".join(keep).strip()
        block = re.sub(r"\n\s*\n", "\n", block)
        block = "\n".join(l.strip() for l in block.splitlines())
        return block, dur, weight

    prompts = []
    durations = []
    weights = []
    for m in re.finditer(rf"{seq_start}\s*(.*?)\s*{seq_end}", text, re.DOTALL | re.IGNORECASE):
        block, dur, weight = _extract(m.group(1))
        if block:
            prompts.append(block)
            durations.append(dur)
            weights.append(weight)

    # ФОЛБЭК по END-маркерам — если END в тексте больше, чем найдено строгих пар —
    # значит, часть START была потеряна моделью.
    end_matches = list(re.finditer(seq_end, text, re.IGNORECASE))
    if len(end_matches) > len(prompts):
        prompts = []
        durations = []
        weights = []
        cut = gm.end() if gm else 0
        for em in end_matches:
            chunk = text[cut:em.start()]
            sm = re.match(rf"\s*{seq_start}\s*", chunk, re.IGNORECASE)
            if sm:
                chunk = chunk[sm.end():]
            block, dur, weight = _extract(chunk)
            if block:
                prompts.append(block)
                durations.append(dur)
                weights.append(weight)
            cut = em.end()

    return global_prompt, prompts, durations, weights


# ══════════════════════════════════════════════════
# CREATE — TripleUpscale (Tripple.json → mod_Tripple.json)
# ══════════════════════════════════════════════════

def _classify_clip_encode(node):
    """Определяет, какое поле промта дублирует CLIPTextEncode-нода."""
    marker = " ".join(
        filter(None, [node.get("title"), *[i.get("label") for i in node.get("inputs", [])]])
    ).upper()
    if "GLOBAL POSITIVE" in marker:
        return "global_positive"
    if "GLOBAL NEGATIVE" in marker:
        return "global_negative"
    if "SCENE PROMPT" in marker or "SZENEN-PROMPT" in marker or "NORMALER SZENEN" in marker:
        return "scene_prompt"
    if "SCENE NEGATIVE" in marker or "SZENEN-NEGATIV" in marker:
        return "scene_negative"
    return None


def create_tripple_workflow(template_path, out_path, prompts, image_path=None,
                            fps=None, frames=None, comfy_input_dir=COMFY_INPUT_DIR):
    """
    Берёт шаблон Tripple.json, вставляет 4 поля промта в ноду «Tripple»
    (+ дубликаты CLIPTextEncode / Primitive fps+кадры / EmptyLTXVLatentVideo),
    копирует стартовое изображение в comfy_input_dir и пишет новый JSON.

    prompts: dict {global_positive, scene_prompt, global_negative, scene_negative}
    Возвращает Path созданного JSON.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)
    with open(template_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    tripple_nodes = find_tripple_nodes(workflow)
    if not tripple_nodes:
        raise ValueError(f"Нода «Tripple» ({TRIPPLE_NODE_TYPE}) не найдена в {template_path.name}")
    node = tripple_nodes[0]
    wv = node["widgets_values"]
    wv[W_GLOBAL_POSITIVE] = as_text(prompts.get("global_positive"))
    wv[W_SCENE_PROMPT]    = as_text(prompts.get("scene_prompt"))
    wv[W_GLOBAL_NEGATIVE] = as_text(prompts.get("global_negative"))
    wv[W_SCENE_NEGATIVE]  = as_text(prompts.get("scene_negative"))

    # fps / кадры
    if fps is not None:
        wv[W_FPS] = int(fps)
    if frames is not None:
        wv[W_FRAMES] = int(frames)

    # Стартовое изображение (I2V) → ComfyUI input + LoadImage-нода
    img_name = None
    if image_path:
        src = Path(image_path)
        if src.exists():
            comfy_input_dir = Path(comfy_input_dir)
            comfy_input_dir.mkdir(parents=True, exist_ok=True)
            img_name = src.name
            dst = comfy_input_dir / img_name
            if dst.resolve() != src.resolve():
                shutil.copy2(src, dst)
            print(f"🖼️  Стартовый кадр → {dst}")

    for n in iter_nodes(workflow):
        ntype = n.get("type") or ""
        title = (n.get("title") or "").lower()
        nwv = n.get("widgets_values") or []
        if ntype == "LoadImage":
            if img_name:
                nwv[0] = img_name
        elif ntype == "CLIPTextEncode":
            field = _classify_clip_encode(n)
            if field and field in prompts and prompts.get(field) is not None:
                if nwv:
                    nwv[0] = as_text(prompts[field])
        elif ntype == "EmptyLTXVLatentVideo" and frames is not None:
            if len(nwv) > 2:
                nwv[2] = int(frames)
        elif ntype == "LTXVEmptyLatentAudio" and frames is not None:
            if nwv:
                nwv[0] = int(frames)
        elif ntype == "PrimitiveInt" and "frame" in title and frames is not None:
            if nwv:
                nwv[0] = int(frames)
        elif ntype == "PrimitiveFloat" and "fps" in title and fps is not None:
            if nwv:
                nwv[0] = int(fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print(f"💾 Tripple-JSON: {out_path}")
    return out_path


# ══════════════════════════════════════════════════
# CREATE — CS LTX Director (SC.json → mod_SC.json)
# ══════════════════════════════════════════════════

def _cs_timeline(global_prompt, segment_prompts, frames, ref_chars, seg_images, subfolder, lengths=None):
    """Собирает timeline_data JSON для LTXDirectorCS.

    lengths — готовые длины сегментов в кадрах (из compute_segment_lengths,
    учитывает разметку LLM). Если не передан — старое равное деление.
    """
    n = len(segment_prompts)
    if not lengths or len(lengths) != n:
        seg_len = max(1, frames // n) if frames and n else 100
        lengths = [seg_len] * n
        if frames and n:
            lengths[-1] = frames - seg_len * (n - 1)
    segments = []
    now_ms = int(time.time() * 1000)
    start = 0
    for i, prompt in enumerate(segment_prompts):
        length = lengths[i]
        seg = {
            "id": f"{now_ms}seg{i}",
            "start": start,
            "length": length,
            "prompt": prompt,
            "type": "text",
            "isAnchor": False,
            "isEndFrame": False,
        }
        if seg_images and i < len(seg_images) and seg_images[i]:
            img_file = seg_images[i]
            seg["type"] = "image"
            seg["imageFile"] = f"{subfolder}/{img_file}" if subfolder else img_file
            seg["imageB64"] = (f"/api/view?filename={img_file}&type=input&subfolder={subfolder}"
                               if subfolder else f"/api/view?filename={img_file}&type=input&subfolder=")
        segments.append(seg)
        start += length

    return {
        "global_prompt": global_prompt,
        "normalDurationFrames": frames,
        "reference_mode": "OFF",
        "characters": ref_chars,
        "segments": segments,
        # остальные ключи шаблона (mainTrackEnabled и пр.) копируются в create_sc_workflow
    }


def create_sc_workflow(template_path, out_path, timelines, fps=24, comfy_input_dir=COMFY_INPUT_DIR,
                       node_type="LTXDirectorCS"):
    """
    Берёт шаблон (SC.json для LTXDirectorCS, LTX_2_5_Director_CStest.json для LTXDirectorCS25),
    обновляет ноды по принципу update_workflow.py (high.json): timeline_data (global_prompt, segments),
    characters (@ref1-3 + VL-описания), widgets_values и properties — синхронно. node_type —
    тип ноды в шаблоне ("LTXDirectorCS" или "LTXDirectorCS25").

    timelines: список из 1 или 2 dict (порядок = порядок таймлайнов слева
               направо на холсте, т.е. timelines[0] → первая по X-координате
               нода LTXDirectorCS, timelines[1] → вторая), каждый:
        {
            "global_prompt":    str,
            "segment_prompts":  [str, ...],
            "ref_images":       [{"name": str, "description": str}, ...] до 3,
            "seg_images":       [str|None, ...] той же длины что segment_prompts, или None,
            "durations":       [float|None, ...] той же длины что segment_prompts — разметка длительности
                                в секундах от LLM (тег [DURATION: X.Xs], см. parse_director_answer); если None или
                                не все заполнены — сегменты делятся поровну (compute_segment_lengths).
            "weights":         [int|None, ...] той же длины что segment_prompts — DURATION_WEIGHT 1-5 от LLM
                                (строка в конце SEQUENCE, вырезается парсером); приоритетнее durations.
            "subfolder":       str,
            "frames":           int | None (если None — берётся из шаблона/30 сек),
        }
    Если передан 1 таймлайн — обновляется только первая (самая левая) нода,
    остальные ноды LTXDirectorCS в шаблоне не трогаются (одиночный режим).
    Возвращает Path созданного JSON.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)
    with open(template_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    cs_nodes = find_nodes(workflow, node_type)
    if not cs_nodes:
        raise ValueError(f"Нода {node_type} не найдена в {template_path.name}")
    # Сопоставляем ноды таймлайнам по X-координате на холсте (слева направо)
    cs_nodes = sorted(cs_nodes, key=lambda n: (n.get("pos") or [0, 0])[0])

    if len(timelines) > len(cs_nodes):
        raise ValueError(
            f"Запрошено {len(timelines)} таймлайнов, а в шаблоне {template_path.name} "
            f"только {len(cs_nodes)} нод(ы) {node_type}"
        )

    fps = int(fps) if fps else 24

    for node, tl_data in zip(cs_nodes, timelines):
        global_prompt   = tl_data.get("global_prompt") or ""
        segment_prompts = tl_data.get("segment_prompts") or []
        ref_images      = tl_data.get("ref_images") or []
        seg_images      = tl_data.get("seg_images")
        subfolder       = tl_data.get("subfolder") or ""
        frames          = tl_data.get("frames")

        wv = node["widgets_values"]
        if frames is None:
            frames = wv[CS_DUR_FR] if len(wv) > CS_DUR_FR and wv[CS_DUR_FR] else frames_for_seconds(30, fps)
        frames = int(frames)
        seconds = round(frames / fps, 2)

        # characters: максимум 3 слота @ref1/@ref2/@ref3; каждый слот = датасет (несколько картинок)
        ref_chars = []
        for r in ref_images[:3]:
            imgs = [i.get("name") or "" for i in (r.get("images") or [])]
            imgs = [f"{subfolder}/{n}" if subfolder else n for n in imgs if n]
            ref_chars.append({
                "images": [{"name": n} for n in imgs],
                "description": (r.get("description") or "").strip(),
            })
        while len(ref_chars) < 3:
            ref_chars.append({"images": [], "description": ""})

        n_seg = len(segment_prompts)
        durations = tl_data.get("durations")
        weights = tl_data.get("weights")
        lengths = compute_segment_lengths(frames, n_seg, durations=durations, weights=weights)
        guide_strength = ",".join(["1.00"] * n_seg)
        local_prompts_str = " | ".join(segment_prompts)

        tl = json.loads(wv[CS_TIMELINE]) if isinstance(wv[CS_TIMELINE], str) else dict(wv[CS_TIMELINE] or {})
        new_tl = _cs_timeline(global_prompt, segment_prompts, frames, ref_chars, seg_images, subfolder, lengths=lengths)
        # сохраняем прочие ключи шаблона (mainTrackEnabled, retake*, analyzeProvider ...)
        for k, v in tl.items():
            if k not in new_tl:
                new_tl[k] = v

        # ── widgets_values ──
        wv[CS_TIMELINE] = json.dumps(new_tl, ensure_ascii=False)
        wv[CS_LOCAL_PROMPTS] = local_prompts_str
        wv[CS_SEG_LENGTHS] = ",".join(str(x) for x in lengths)
        wv[CS_DUR_FR] = frames
        wv[CS_END_FR] = frames
        wv[CS_START_FR] = 0
        wv[CS_DUR_SEC] = seconds
        wv[CS_END_SEC] = seconds
        wv[CS_START_SEC] = 0
        if len(wv) > CS_FPS:
            wv[CS_FPS] = fps
        if len(wv) > 10 and isinstance(wv[10], str):
            wv[10] = guide_strength

        # ── properties — синхронизация (критично для LTXDirectorCS) ──
        props = node.get("properties")
        if isinstance(props, dict):
            props["global_prompt"] = global_prompt
            props["local_prompts"] = local_prompts_str
            props["segment_lengths"] = ",".join(str(x) for x in lengths)
            props["timeline_data"] = json.dumps(new_tl, ensure_ascii=False)
            props["frame_rate"] = fps
            props["start_second"] = 0
            props["end_second"] = seconds
            props["duration_seconds"] = seconds
            props["start_frame"] = 0
            props["end_frame"] = frames
            props["duration_frames"] = frames
            props["guide_strength"] = guide_strength

        pos_x = (node.get("pos") or [0])[0]
        print(f"⚙️  {node_type} id={node.get('id')} (x={pos_x:.0f}): {n_seg} сегментов, {frames} кадров @ {fps} fps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print(f"💾 CS-JSON: {out_path}")
    return out_path


# ══════════════════════════════════════════════════
# CREATE — LTX 2.5 First/Last Frame (первый + последний кадр)
# ══════════════════════════════════════════════════

def sigma_curve(steps, start=1.0, exp=3.0):
    """Кривая ManualSigmas start→0: sigma(t) = start * (1 - t^exp)."""
    steps = max(2, int(steps))
    out = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        out.append(round(start * (1.0 - t ** exp), 5))
    out[-1] = 0.0
    return out


def compute_sigmas(duration, fps=24, min_steps=8, max_steps=24, steps=None):
    """
    Кривая ManualSigmas 1.0 → 0.0 (по образцу шаблона: ~1 - t^3).
    steps — число шагов (решение LLM через SIGMA_STEPS); если None —
    шагов тем больше, чем длиннее ролик: clamp(round(duration*0.8), min, max).
    """
    if steps is None:
        steps = min(max_steps, max(min_steps, round(duration * 0.8)))
    return sigma_curve(steps)


def compute_t2v_sigmas(duration, fps=24, high_steps=None):
    """Две кривые ManualSigmas (low-res / high-res) для Text-to-Video.
    high_steps — число шагов high-стадии (решение LLM через SIGMA_STEPS);
    low = clamp(round(high*0.45), 3, 10). По умолчанию для 16s → 4/9, как в шаблоне
    (low: старт 0.85, эксп 1.7; high: 1 - t^3)."""
    high = high_steps if high_steps else min(20, max(8, round(duration * 0.56)))
    low = min(10, max(3, round(high * 0.45)))
    return sigma_curve(low, 0.85, 1.7), sigma_curve(high, 1.0, 3.0)


def parse_sigma_steps(text):
    """
    Извлекает строку 'SIGMA_STEPS: N' (решение LLM о числе шагов денойза)
    из промта и ВЫРЕЗАЕТ её. Возвращает (clean_text, steps|None).
    """
    if not text:
        return text, None
    lines = text.splitlines()
    keep = []
    steps = None
    for ln in lines:
        m = re.match(r"(?i)^\s*SIGMA\s*_?\s*STEPS\s*:\s*(\d{1,2})\s*$", ln.strip())
        if m:
            s = int(m.group(1))
            if 4 <= s <= 40:
                steps = s
            continue
        keep.append(ln)
    return "\n".join(keep).strip(), steps


def create_ltx25_workflow(template_path, out_path, prompt_text,
                          first_image=None, last_image=None,
                          duration=None, fps=24, sigma_values=None,
                          comfy_input_dir=COMFY_INPUT_DIR):
    """
    Берёт шаблон «LTX 2.5 First and Last Frame Video Generation .json» и заполняет:
      - LoadImage (первый по X = START-кадр, второй = END-кадр) — картинки из UI
      - PrimitiveStringMultiline «Prompt» + пустой CLIPTextEncode — промт
      - PrimitiveInt «Duration» (секунды) и «Frame Rate(int)»
      - ManualSigmas — кривая 1→0 (шаги зависят от длительности)
    Кадры копируются в comfy_input_dir (LoadImage берёт имя из корня input).
    Возвращает Path созданного JSON.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)
    with open(template_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    prompt_text = as_text(prompt_text)

    # ── LoadImage: первый по X = START-кадр, второй = END-кадр ──
    load_imgs = sorted(find_nodes(workflow, "LoadImage"), key=lambda n: (n.get("pos") or [0, 0])[0])
    for node, img in zip(load_imgs[:2], [first_image, last_image]):
        if img and Path(img).exists():
            src = Path(img)
            comfy_input_dir = Path(comfy_input_dir)
            comfy_input_dir.mkdir(parents=True, exist_ok=True)
            dst = comfy_input_dir / src.name
            if dst.resolve() != src.resolve():
                shutil.copy2(src, dst)
            node["widgets_values"][0] = src.name
            print(f"🖼️  {src.name} → LoadImage (id {node.get('id')})")

    # ── промт: PrimitiveStringMultiline «Prompt» + пустой CLIPTextEncode ──
    for node in find_nodes(workflow, "PrimitiveStringMultiline"):
        if (node.get("title") or "").lower() == "prompt":
            node["widgets_values"][0] = prompt_text
    for node in find_nodes(workflow, "CLIPTextEncode"):
        wv = node.get("widgets_values") or []
        if not wv or not str(wv[0] or "").strip():
            wv[0] = prompt_text  # пустой = позитивный промт

    # ── длительность (секунды) и FPS ──
    if duration:
        for node in find_nodes(workflow, "PrimitiveInt"):
            title = (node.get("title") or "").lower()
            if title == "duration":
                node["widgets_values"][0] = int(duration)
            elif "frame rate" in title:
                node["widgets_values"][0] = int(fps)

    # ── ManualSigmas: кривая 1→0, шаги от длительности ──
    sigmas = sigma_values if sigma_values else compute_sigmas(duration or 15, int(fps) or 24)
    for node in find_nodes(workflow, "ManualSigmas"):
        node["widgets_values"][0] = ", ".join(str(x) for x in sigmas)
        print(f"⚙️  ManualSigmas (id {node.get('id')}): {len(sigmas)} шагов")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print(f"💾 LTX-2.5-JSON: {out_path}")
    return out_path


def create_t2v_workflow(template_path, out_path, prompt_text,
                        duration=None, fps=24, sigmas_low=None, sigmas_high=None):
    """
    Text-to-Video (Fast LTX2.5): заполняет
      - PrimitiveStringMultiline «Prompt» + пустой CLIPTextEncode — промт
      - PrimitiveInt «Duration» (секунды) и «Frame Rate»
      - ManualSigmas ×2: первый по X = low-res (кривая 0.85→0), второй = high-res (1→0)
    Кадры считает ComfyMathExpression (duration*fps+1) — не трогаем.
    Возвращает Path созданного JSON.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)
    with open(template_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    prompt_text = as_text(prompt_text)

    # ── промт: PrimitiveStringMultiline «Prompt» + пустой CLIPTextEncode ──
    for node in find_nodes(workflow, "PrimitiveStringMultiline"):
        if (node.get("title") or "").lower() == "prompt":
            node["widgets_values"][0] = prompt_text
    for node in find_nodes(workflow, "CLIPTextEncode"):
        wv = node.get("widgets_values") or []
        if not wv or not str(wv[0] or "").strip():
            wv[0] = prompt_text  # пустой = позитивный промт

    # ── длительность (секунды) и FPS ──
    if duration:
        for node in find_nodes(workflow, "PrimitiveInt"):
            title = (node.get("title") or "").lower()
            if title == "duration":
                node["widgets_values"][0] = int(duration)
            elif "frame rate" in title:
                node["widgets_values"][0] = int(fps)

    # ── ManualSigmas ×2: low / high — определяем по числу шагов в шаблоне (меньше = low) ──
    sig_low = sigmas_low if sigmas_low else compute_t2v_sigmas(duration or 16, int(fps) or 24)[0]
    sig_high = sigmas_high if sigmas_high else compute_t2v_sigmas(duration or 16, int(fps) or 24)[1]

    def _n_vals(node):
        try:
            return len([x for x in str((node.get("widgets_values") or [" "])[0]).split(",") if x.strip()])
        except Exception:
            return 0

    ms_nodes = sorted(find_nodes(workflow, "ManualSigmas"),
                      key=lambda n: (_n_vals(n), (n.get("pos") or [0, 0])[0]))
    for idx, node in enumerate(ms_nodes[:2]):
        vals = sig_low if idx == 0 else sig_high
        node["widgets_values"][0] = ", ".join(str(x) for x in vals)
        print(f"⚙️  ManualSigmas (id {node.get('id')}): {len(vals)} шагов")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print(f"💾 T2V-JSON: {out_path}")
    return out_path


# ══════════════════════════════════════════════════
# CREATE — MiniMax H3 Reference-to-Video (Minimax_reference_pictures.json)
# ══════════════════════════════════════════════════

MINIMAX_REF_NODE_TYPE = "MiniMaxH3ReferenceToVideo"
MINIMAX_MAX_PICTURES = 9


def parse_minimax_answer(text):
    """
    Разбирает ответ LLM в формате minimaxh3SP.txt:
        VIDEO: ... SOUND: ... MUSIC: ...
    Возвращает (merged_prompt, parts) — merged_prompt это ОДНА строка для ноды промта
    MiniMax H3 (видео-часть, затем звук и музыка), parts = {video, sound, music}.
    Если заголовков нет — весь текст считается видео-частью.
    """
    text = (text or "").lstrip("\ufeff").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    head = re.compile(r"(?im)^[ \t*#]*(VIDEO|SOUND|MUSIC)[ \t*]*:[ \t*]*")
    marks = list(head.finditer(text))
    parts = {"video": "", "sound": "", "music": ""}
    if not marks:
        parts["video"] = text
    else:
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            parts[m.group(1).lower()] = text[m.end():end].strip()
        # текст ДО первого заголовка (модель потеряла VIDEO:) — это видео-часть
        pre = text[:marks[0].start()].strip()
        if pre and not parts["video"]:
            parts["video"] = pre
    merged = parts["video"].strip()
    if parts["sound"].strip():
        merged += "\n\nSOUND DESIGN: " + re.sub(r"\s*\n\s*", " ", parts["sound"].strip())
    music = parts["music"].strip()
    if music and music.lower().strip(" .") != "none":
        merged += "\n\nMUSIC: " + re.sub(r"\s*\n\s*", " ", music)
    return merged.strip(), parts


def _link_origin(workflow, link_id):
    """(origin_node_id, origin_slot) для link_id; поддержка списочного и словарного формата links."""
    for l in workflow.get("links", []) or []:
        if isinstance(l, (list, tuple)):
            if l and l[0] == link_id:
                return l[1], l[2]
        elif isinstance(l, dict) and l.get("id") == link_id:
            return l.get("origin_id"), l.get("origin_slot")
    return None, None


def _input_origin_node(workflow, nodes_by_id, node, input_name):
    """Нода, подключённая ко входу input_name ноды node (или None)."""
    for inp in node.get("inputs", []) or []:
        if inp.get("name") == input_name and inp.get("link") is not None:
            oid, _ = _link_origin(workflow, inp["link"])
            return nodes_by_id.get(oid)
    return None


def _prompt_widget_index(onode, widgets_values, out_slot=None):
    """Позиция текстового виджета «Prompt» в widgets_values connected-ноды.

    Ищем не «самую длинную строку», а виджет по роли значения (value) в схеме
    ноды: имя виджета берётся из properties/widgets_values_named (или из label
    связанного input), затем его позиционный индекс находится в widgets_values.
    Возвращает индекс в widgets_values или None.
    """
    if not isinstance(widgets_values, list):
        return None

    def _norm(s):
        return "".join(str(s or "").lower().split())

    # 1) имя value-виджета из подписанной схемы свойств (widgets_values_named /
    #    cnr_properties): сопоставляем значение с widgets_values по содержимому
    named = {}
    props = onode.get("properties") or {}
    for src in (onode.get("widgets_values_named"),
                props.get("widget_values"),
                props.get("cnr_properties")):
        if isinstance(src, dict):
            named.update(src)
    if named:
        # candidate widget names tied to the prompt role
        role_names = [k for k in named
                      if _norm(k) in ("value", "prompt", "inputvalue")]
        for name in role_names:
            val = named[name]
            if isinstance(val, str):
                for i, v in enumerate(widgets_values):
                    if v == val:
                        return i

    # 2) label связанного input со ссылкой на промтовый вывод (роль value/Prompt)
    inputs = onode.get("inputs", []) or []
    outputs = onode.get("outputs", []) or []
    slot = out_slot if (out_slot is not None and 0 <= out_slot < len(outputs)) else None
    if slot is None:
        slot = next((k for k, e in enumerate(outputs)
                     if isinstance(e, dict) and (_norm(e.get("label")) in ("prompt", "value")
                                                 or _norm(e.get("name")) in ("prompt", "value"))), None)
    want_label = None
    if slot is not None:
        e = outputs[slot]
        want_label = _norm(e.get("label") or e.get("name"))
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        lbl = _norm(inp.get("label") or inp.get("name"))
        wid = _norm((inp.get("widget") or {}).get("name")) if isinstance(inp.get("widget"), dict) else ""
        if want_label is not None and lbl != want_label and wid != want_label:
            continue
        # этот input — value-роль промта; его widget-имя может быть в type/name
        for cand in (wid, lbl, _norm(inp.get("name"))):
            if cand in ("value", "prompt"):
                # позиция среди string-виджетов: сколько string-входов до него
                preceding = sum(1 for x in inputs[:inputs.index(inp)]
                                if isinstance(x, dict) and str(x.get("type", "")).upper() == "STRING")
                str_positions = [i for i, v in enumerate(widgets_values) if isinstance(v, str)]
                if preceding < len(str_positions):
                    return str_positions[preceding]
                if str_positions:
                    return str_positions[0]
                return None

    # 3) последний вариант: Prompt-выход с типом STRING → первый string-виджет
    if slot is not None and str(outputs[slot].get("type", "")).upper() == "STRING":
        preceding = sum(1 for e in outputs[:slot]
                        if isinstance(e, dict) and str(e.get("type", "")).upper() == "STRING")
        str_positions = [i for i, v in enumerate(widgets_values) if isinstance(v, str)]
        if preceding < len(str_positions):
            return str_positions[preceding]
    return None


def _copy_to_input(src, comfy_input_dir):
    """Копирует файл в ComfyUI input, возвращает имя файла (или None, если файла нет)."""
    if not src:
        return None
    src = Path(src)
    if not src.exists():
        print(f"⚠️  Файл не найден: {src}")
        return None
    comfy_input_dir = Path(comfy_input_dir)
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    dst = comfy_input_dir / src.name
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    return src.name


def create_minimax_ref_workflow(template_path, out_path, prompt_text, images=None,
                                duration=None, audio=None, video=None,
                                aspect=None, megapixels=None,
                                comfy_input_dir=COMFY_INPUT_DIR):
    """
    Берёт шаблон Minimax_reference_pictures.json и заполняет его.
    Ноды находятся ПО СВЯЗЯМ ноды MiniMaxH3ReferenceToVideo (не по id и не по X-позиции):
      - вход prompt           → PrimitiveStringMultiline с промтом
      - ref_images.ref_image_N → LoadImage (слот N = <Picture N+1> в промте)
      - length ← ComfyMathExpression ← values.a ← PrimitiveFloat «Duration» (секунды)
      - ref_audios.ref_audio_0 → LoadAudio, ref_videos.ref_video_0 → VHS_LoadVideo
      - width/height ← subgraph ResolutionSelector (aspect / megapixels), опционально
    images: список путей (до 9). Заполненные слоты включаются (mode 0), остальные LoadImage
    отключаются (bypass, mode 4). audio/video: если не переданы — ноды отключаются.
    Возвращает Path созданного JSON.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)
    with open(template_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # только верхний уровень: внутри subgraph есть свои PrimitiveStringMultiline/PrimitiveFloat
    nodes_by_id = {n.get("id"): n for n in workflow.get("nodes", [])}
    mains = [n for n in workflow.get("nodes", []) if n.get("type") == MINIMAX_REF_NODE_TYPE]
    if not mains:
        raise ValueError(f"Нода {MINIMAX_REF_NODE_TYPE} не найдена в {template_path.name}")
    main = mains[0]

    # ── промт ──
    pnode = _input_origin_node(workflow, nodes_by_id, main, "prompt")
    if pnode is None or not pnode.get("widgets_values"):
        raise ValueError("К входу prompt ноды MiniMaxH3ReferenceToVideo не подключена нода с текстом")
    pnode["widgets_values"][0] = as_text(prompt_text)

    # если у текстовой ноды есть СВОЙ вход value, подключённый к другой ноде
    # (например, subgraph-обёртка «Prompt/Duration/Width/Height»), ComfyUI при загрузке
    # перерисует виджет из связи и новый текст пропадёт. Разрываем эту связь —
    # тогда значение виджета используется как есть. Заодно пишем новый текст
    # в ноду-источник, чтобы старый промт нигде не остался.
    for inp in pnode.get("inputs", []) or []:
        if inp.get("name") == "value" and inp.get("link") is not None:
            lid = inp["link"]
            oid, _ = _link_origin(workflow, lid)
            onode = nodes_by_id.get(oid) if oid is not None else None
            if onode is not None:
                prompt_slot = None
                for outp in onode.get("outputs", []) or []:
                    if isinstance(outp.get("links"), list) and lid in outp["links"]:
                        outp["links"].remove(lid)
                        # слот вывода, отдающий значение по связи (по позиции в outputs)
                        if prompt_slot is None:
                            try:
                                prompt_slot = onode["outputs"].index(outp)
                            except ValueError:
                                pass
                owv = onode.get("widgets_values")
                if isinstance(owv, list):
                    # позицию промта берём из схемы connected-ноды: виджет,
                    # соответствующий значению «Prompt» (роль value / label Prompt),
                    # а не самый длинный текст
                    idx = _prompt_widget_index(onode, owv, prompt_slot)
                    if idx is not None:
                        owv[idx] = as_text(prompt_text)
                        print(f"📝 Промт записан также в connected-ноду id {onode.get('id')} (позиция {idx})")
            workflow["links"] = [l for l in workflow.get("links", []) or []
                                 if not (isinstance(l, (list, tuple)) and l and l[0] == lid)
                                 and not (isinstance(l, dict) and l.get("id") == lid)]
            inp["link"] = None
            print(f"🔗 Убран входящий link на ноде промта (id {pnode.get('id')}) — "
                  f"текст берётся из виджета")

    # ── картинки ──
    images = [p for p in (images or []) if p][:MINIMAX_MAX_PICTURES]
    used = 0
    for slot in range(MINIMAX_MAX_PICTURES):
        node = _input_origin_node(workflow, nodes_by_id, main, f"ref_images.ref_image_{slot}")
        if node is None:
            continue
        name = _copy_to_input(images[slot], comfy_input_dir) if slot < len(images) else None
        if name:
            node["widgets_values"][0] = name
            node["mode"] = 0
            used += 1
            print(f"🖼️  <Picture {slot + 1}> = {name} (LoadImage id {node.get('id')})")
        else:
            node["mode"] = 4
    if images and used < len(images):
        print(f"⚠️  Использовано {used} из {len(images)} картинок (нет файла или нет слота)")

    # ── длительность (секунды) через ComfyMathExpression → length ──
    if duration:
        math_node = _input_origin_node(workflow, nodes_by_id, main, "length")
        dnode = None
        if math_node is not None:
            dnode = _input_origin_node(workflow, nodes_by_id, math_node, "values.a")
        if dnode is None:
            for n in workflow.get("nodes", []):
                if n.get("type") == "PrimitiveFloat" and "duration" in (n.get("title") or "").lower():
                    dnode = n
                    break
        if dnode is not None and dnode.get("widgets_values"):
            dnode["widgets_values"][0] = float(duration)
            print(f"⏱️  Duration = {float(duration)} с (нода id {dnode.get('id')})")
        else:
            print("⚠️  Нода Duration не найдена — длительность не записана")

    # ── референс-аудио ──
    anode = _input_origin_node(workflow, nodes_by_id, main, "ref_audios.ref_audio_0")
    if anode is not None:
        aname = _copy_to_input(audio, comfy_input_dir)
        if aname:
            anode["widgets_values"][0] = aname
            anode["mode"] = 0
            print(f"🔊 <Audio 1> = {aname}")
        else:
            anode["mode"] = 4

    # ── референс-видео ──
    vnode = _input_origin_node(workflow, nodes_by_id, main, "ref_videos.ref_video_0")
    if vnode is not None:
        vname = _copy_to_input(video, comfy_input_dir)
        if vname:
            wv = vnode.get("widgets_values")
            if isinstance(wv, dict):
                wv["video"] = vname
                prev = wv.get("videopreview")
                if isinstance(prev, dict) and isinstance(prev.get("params"), dict):
                    prev["params"]["filename"] = vname
            vnode["mode"] = 0
            print(f"🎞️  <Video 1> = {vname}")
        else:
            vnode["mode"] = 4

    # ── разрешение (subgraph с ResolutionSelector) ──
    if aspect or megapixels:
        rnode = _input_origin_node(workflow, nodes_by_id, main, "width")
        if rnode is not None and isinstance(rnode.get("widgets_values"), list):
            if aspect:
                rnode["widgets_values"][0] = aspect
            if megapixels and len(rnode["widgets_values"]) > 1:
                rnode["widgets_values"][1] = float(megapixels)
            print(f"📐 Разрешение: {aspect or '—'} / {megapixels or '—'} MP")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print(f"💾 MiniMax-Ref-JSON: {out_path}")
    return out_path


# ══════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════

def _cli_create(argv):
    """python tripple.py create --out <path> --image <path> [--fps 24] [--frames N]
    [--global-positive F] [--scene-prompt F] [--global-negative F] [--scene-negative F]
    [--template <Tripple.json>] [--comfy-input <dir>]"""
    import argparse
    ap = argparse.ArgumentParser(prog="tripple.py create")
    ap.add_argument("--template", default=str(DEFAULT_JSON))
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--global-positive", default=None)
    ap.add_argument("--scene-prompt", default=None)
    ap.add_argument("--global-negative", default=None)
    ap.add_argument("--scene-negative", default=None)
    ap.add_argument("--flux-prompt", default=None, help="промт стартового кадра для FLUX 2 Klein → flux_start_image.txt")
    ap.add_argument("--comfy-input", default=str(COMFY_INPUT_DIR))
    a = ap.parse_args(argv)

    if a.global_positive is None or a.scene_prompt is None:
        print("❌ Нужны --global-positive и --scene-prompt (файлы или текст).")
        sys.exit(1)

    def _read(v):
        if v is None:
            return ""
        p = Path(v)
        return p.read_text(encoding="utf-8") if p.exists() else v

    prompts = {
        "global_positive": _read(a.global_positive),
        "scene_prompt":    _read(a.scene_prompt),
        "global_negative": _read(a.global_negative),
        "scene_negative":  _read(a.scene_negative),
    }
    create_tripple_workflow(
        template_path=a.template, out_path=a.out, prompts=prompts,
        image_path=a.image, fps=a.fps, frames=a.frames, comfy_input_dir=a.comfy_input,
    )
    if a.flux_prompt:
        flux_text = _read(a.flux_prompt).strip()
        if flux_text:
            flux_path = Path(a.out).parent / "flux_start_image.txt"
            flux_path.write_text(flux_text + "\n", encoding="utf-8")
            print(f"🖼️  FLUX-Startbild-Prompt (Klein) → {flux_path}")


def _cli_create_sc(argv):
    """python tripple.py create-sc --out <path> [--global-prompt F] [--prompts F]
    [--ref1 name:desc] ... [--seg-image N] ... [--subfolder S] [--fps 24] [--frames N]
    [--template <SC.json>] [--comfy-input <dir>]"""
    import argparse
    ap = argparse.ArgumentParser(prog="tripple.py create-sc")
    ap.add_argument("--template", default=str(DEFAULT_SC))
    ap.add_argument("--out", required=True)
    ap.add_argument("--global-prompt", default=None)
    ap.add_argument("--prompts", default=None, help="файл с ---SEQUENCE_N--- блоками или строка")
    ap.add_argument("--ref", action="append", default=[], help="name:description (до 3)")
    ap.add_argument("--seg-image", action="append", default=[], help="имя файла сегмента (по индексу)")
    ap.add_argument("--subfolder", default="")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--comfy-input", default=str(COMFY_INPUT_DIR))
    a = ap.parse_args(argv)

    global_prompt = None
    segment_prompts = []
    segment_durations = None
    if a.prompts:
        txt = Path(a.prompts).read_text(encoding="utf-8") if Path(a.prompts).exists() else a.prompts
        gp, seqs, durs, wts = parse_director_answer(txt)
        if not seqs and not a.global_prompt:
            print("❌ Не удалось разобрать ---SEQUENCE--- блоки из --prompts.")
            sys.exit(1)
        global_prompt = global_prompt or gp
        segment_prompts = seqs
        segment_durations = durs
        segment_weights = wts
    if a.global_prompt:
        global_prompt = (Path(a.global_prompt).read_text(encoding="utf-8")
                         if Path(a.global_prompt).exists() else a.global_prompt)
    if not segment_prompts:
        print("❌ Нет сегментных промтов (--prompts с ---SEQUENCE--- блоками).")
        sys.exit(1)

    ref_images = []
    for r in a.ref:
        name, _, desc = r.partition(":")
        name = name.strip()
        ref_images.append({"images": [{"name": name}] if name else [], "description": desc.strip()})

    seg_images = a.seg_image or None
    create_sc_workflow(
        template_path=a.template, out_path=a.out,
        timelines=[{
            "global_prompt":   global_prompt or "",
            "segment_prompts": segment_prompts,
            "ref_images":      ref_images,
            "seg_images":      seg_images,
            "durations":       segment_durations,
            "weights":         segment_weights,
            "subfolder":       a.subfolder,
            "frames":          a.frames,
        }],
        fps=a.fps, comfy_input_dir=a.comfy_input,
    )


def _cli_create_ltx25(argv):
    """python tripple.py create-ltx25 --out <path> --prompt <файл|текст>
    [--first <img>] [--last <img>] [--duration 15] [--fps 24] [--sigma N,...]
    [--template <LTX 2.5 ...json>] [--comfy-input <dir>]"""
    import argparse
    ap = argparse.ArgumentParser(prog="tripple.py create-ltx25")
    ap.add_argument("--template", default=str(DEFAULT_LTX25))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True, help="файл с промтом или строка")
    ap.add_argument("--first", default=None)
    ap.add_argument("--last", default=None)
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--sigma", default=None, help="CSV сигам, напр. '1.0,0.5,0.0'")
    ap.add_argument("--comfy-input", default=str(COMFY_INPUT_DIR))
    a = ap.parse_args(argv)

    p = Path(a.prompt)
    prompt_text = p.read_text(encoding="utf-8") if p.exists() else a.prompt

    sigmas = None
    if a.sigma:
        sigmas = [float(x.strip()) for x in a.sigma.split(",") if x.strip()]

    create_ltx25_workflow(
        template_path=a.template, out_path=a.out, prompt_text=prompt_text,
        first_image=a.first, last_image=a.last,
        duration=a.duration, fps=a.fps, sigma_values=sigmas,
        comfy_input_dir=a.comfy_input,
    )


def _cli_create_t2v(argv):
    """python tripple.py create-t2v --out <path> --prompt <файл|текст>
    [--duration 16] [--fps 24] [--sigma-low N,...] [--sigma-high N,...]
    [--template <Text-to-Video ...json>]"""
    import argparse
    ap = argparse.ArgumentParser(prog="tripple.py create-t2v")
    ap.add_argument("--template", default=str(DEFAULT_T2V))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True, help="файл с промтом или строка")
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--sigma-low", default=None, help="CSV сигам low-res")
    ap.add_argument("--sigma-high", default=None, help="CSV сигам high-res")
    a = ap.parse_args(argv)

    p = Path(a.prompt)
    prompt_text = p.read_text(encoding="utf-8") if p.exists() else a.prompt

    def _csv(s):
        return [float(x.strip()) for x in s.split(",") if x.strip()] if s else None

    create_t2v_workflow(
        template_path=a.template, out_path=a.out, prompt_text=prompt_text,
        duration=a.duration, fps=a.fps,
        sigmas_low=_csv(a.sigma_low), sigmas_high=_csv(a.sigma_high),
    )


def _cli_create_minimax_ref(argv):
    """python tripple.py create-minimax-ref --out <path> --prompt <файл|текст>
    [--image P]... (до 9) [--duration 10] [--audio P] [--video P]
    [--aspect "9:16 (Portrait Widescreen)"] [--megapixels 0.1] [--template <json>]
    Промт может быть в формате VIDEO:/SOUND:/MUSIC: (объединится в одну строку)."""
    import argparse
    ap = argparse.ArgumentParser(prog="tripple.py create-minimax-ref")
    ap.add_argument("--template", default=str(DEFAULT_MINIMAX_REF))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True, help="файл с промтом или строка")
    ap.add_argument("--image", action="append", default=[], help="картинка-референс (повторять до 9 раз)")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--aspect", default=None)
    ap.add_argument("--megapixels", type=float, default=None)
    ap.add_argument("--comfy-input", default=str(COMFY_INPUT_DIR))
    a = ap.parse_args(argv)

    p = Path(a.prompt)
    raw = p.read_text(encoding="utf-8-sig") if p.exists() else a.prompt
    merged, _parts = parse_minimax_answer(raw)
    create_minimax_ref_workflow(
        template_path=a.template, out_path=a.out, prompt_text=merged,
        images=a.image, duration=a.duration, audio=a.audio, video=a.video,
        aspect=a.aspect, megapixels=a.megapixels, comfy_input_dir=a.comfy_input,
    )


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("extract", "extract-main"):
        extract_main(Path(argv[1]) if len(argv) > 1 else DEFAULT_JSON)
    elif argv[0] == "create":
        _cli_create(argv[1:])
    elif argv[0] == "create-sc":
        _cli_create_sc(argv[1:])
    elif argv[0] == "create-ltx25":
        _cli_create_ltx25(argv[1:])
    elif argv[0] == "create-t2v":
        _cli_create_t2v(argv[1:])
    elif argv[0] == "create-minimax-ref":
        _cli_create_minimax_ref(argv[1:])
    elif argv[0] in ("-h", "--help", "help"):
        print(__doc__)
    else:
        # обратная совместимость: python tripple.py <json>
        extract_main(Path(argv[0]))


if __name__ == "__main__":
    main()
