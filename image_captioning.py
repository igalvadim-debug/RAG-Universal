"""
image_captioning.py — Florence-2 мульти-кадровый генератор промптов для LTX 2.3.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import gc
import io
import os
import re

FLORENCE_DIR = Path(__file__).parent / "florence-2-large"
def _build_timeline_labels(n_frames: int) -> list[str]:
    """Build evenly-spaced percentage labels from 0% to 100%."""
    if n_frames <= 1:
        return ["[0%]"]
    labels = []
    for i in range(n_frames):
        pct = round(i * 100 / (n_frames - 1))
        labels.append(f"[{pct}%]")
    return labels


# ──────────────────────────────────────────────
# NOTIZEN-FILTER — trennt LLM-Meta-Kommentare vom eigentlichen Prompt-Text
# ──────────────────────────────────────────────

_NOTE_PATTERNS = [
    r"(?i)^\s*(note|hinweis|remark|disclaimer)\s*[:\-]",
    r"(?i)\bwithout the (original|actual) (image )?descriptions?\b",
    r"(?i)\bfor demonstration purposes\b",
    r"(?i)\bthis (is|remains) speculative\b",
    r"(?i)\bassuming there (is|are)\b",
    r"(?i)\bcannot be provided\b",
    r"(?i)\bas an ai\b",
    r"(?i)\bi cannot\b",
    r"(?i)\bhowever,? without\b",
    r"(?i)\bplease (provide|note)\b",
    r"(?i)^\s*\(.*(placeholder|example|assum).*\)\s*$",
]


def _split_notes(text: str) -> tuple[str, str]:
    """
    Trennt LLM-Meta-Kommentare (Vorbehalte, Erklaerungen, Platzhalter-Hinweise)
    vom eigentlichen Prompt-Text.
    Gibt (bereinigter_text, notizen) zurueck.
    """
    if not text:
        return text, ""
    lines = text.split("\n")
    clean_lines = []
    note_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append(line)
            continue
        is_note = any(re.search(pat, stripped) for pat in _NOTE_PATTERNS)
        if is_note:
            note_lines.append(stripped)
        else:
            clean_lines.append(line)
    clean = "\n".join(clean_lines).strip()
    notes = "\n".join(note_lines).strip()
    return clean, notes


REWRITE_SYSTEM = (
    "You are an expert LTX Video 2.3 prompt engineer telling a VIDEO STORY frame by frame.\n"
    "The SCENARIO describes a SEQUENCE of events that unfold OVER TIME.\n"
    "You are writing ONLY ONE KEYFRAME — a single moment in that sequence.\n\n"
    "You receive:\n"
    "  - SCENARIO: the FULL narrative from beginning to end.\n"
    "  - POSITION: which moment you are writing (opening, middle, climax, end).\n"
    "  - IMAGE: visual reference for THIS keyframe (setting, subject, lighting) — if available.\n\n"
    "TEMPORAL DISCIPLINE — THE MOST IMPORTANT RULE:\n"
    "The scenario says what happens FIRST, THEN what happens NEXT, THEN what happens LAST.\n"
    "If your POSITION is 'opening / start': ONLY describe the FIRST event. NOTHING later.\n"
    "If your POSITION is 'midpoint': ONLY describe the middle event. NOT the beginning, NOT the end.\n"
    "If your POSITION is 'closing / end': ONLY describe the final event.\n"
    "NEVER pack multiple events from different parts of the scenario into one keyframe.\n"
    "Example: Scenario = 'man works out, THEN Arab enters, THEN man is shocked'\n"
    "  - At OPENING: ONLY 'man works out calmly'. No Arab. No shock.\n"
    "  - At MIDPOINT: ONLY 'Arab enters shouting'. No shock yet.\n"
    "  - At END: ONLY 'man is shocked'. Arab already there.\n\n"
    "Rules:\n"
    "1. Start with camera movement/position (e.g. 'Slow dolly-in tracking shot', 'Low-angle orbit around the subject', 'Steadycam push-in', 'Whip pan reveals', 'Crane-down establishing'). NEVER start with 'Static'.\n"
    "2. Describe ONLY the event that belongs to THIS POSITION. Use the IMAGE for visual details if provided; "
    "otherwise invent plausible, consistent visual details that match the SCENARIO.\n"
    "3. Include lighting, color grading, and atmosphere.\n"
    "4. Audio is MANDATORY — always include ONE separate, clearly labeled sentence starting with "
    "'Audio: ' describing the concrete sound design for THIS moment (e.g. 'Audio: calm ambient room "
    "tone, distant birdsong.'). Never omit it, never bury it inside another sentence.\n"
    "5. Output ONLY the prompt text — no labels besides the required 'Audio:' sentence, no markdown, "
    "no explanations, no disclaimers, no meta-commentary about missing information.\n"
    "6. Length: 2-4 sentences maximum (the 'Audio:' sentence counts as one of them).\n"
    "7. Language: English only."
)

INTERPOLATION_SYSTEM = (
    "You are an expert LTX Video 2.3 prompt engineer filling GAPS in a video story.\n"
    "A gap is the TRANSITION between two keyframes — something CHANGES here.\n\n"
    "You receive:\n"
    "  - SCENARIO: the FULL narrative from beginning to end.\n"
    "  - PREVIOUS keyframe prompt: what is happening BEFORE this gap.\n"
    "  - NEXT keyframe prompt: what is happening AFTER this gap.\n\n"
    "Your job: Write the TRANSITION — what CHANGES between previous and next.\n"
    "Look at the SCENARIO: what event happens BETWEEN these two moments?\n"
    "The transition should show the CAUSE of the change, not the result.\n\n"
    "TEMPORAL DISCIPLINE:\n"
    "If PREVIOUS shows 'man working out' and NEXT shows 'Arab shouting':\n"
    "  → Write 'Arab ENTERS the gym, APPROACHING, phone in hand' (the transition).\n"
    "  → Do NOT write 'Arab is already shouting' (that's NEXT, not the gap).\n"
    "If PREVIOUS shows 'Arab shouting' and NEXT shows 'man shocked':\n"
    "  → Write 'man's expression SHIFTS, eyes widen, body FREEZES' (the reaction beginning).\n"
    "  → Do NOT write 'man is completely shocked' (that's NEXT, not the gap).\n\n"
    "Rules:\n"
    "1. Start with camera movement (e.g. 'Whip pan reveals', 'Slow dolly captures the approach').\n"
    "2. Describe the IN-BETWEEN event — the moment of CHANGE, not the before or after state.\n"
    "3. Maintain visual continuity with neighboring frames (lighting, color, setting).\n"
    "4. Audio is MANDATORY — always include ONE separate, clearly labeled sentence starting with "
    "'Audio: ' describing how the sound TRANSITIONS here (a sound begins, fades, or shifts). "
    "Never omit it.\n"
    "5. Output ONLY the prompt text — no labels besides the required 'Audio:' sentence, no markdown, "
    "no explanations, no disclaimers.\n"
    "6. Length: 2-4 sentences maximum (the 'Audio:' sentence counts as one of them).\n"
    "7. Language: English only."
)

GLOBAL_PROMPT_SYSTEM = (
    "You are an expert LTX Video 2.3 prompt engineer. "
    "You receive a video SCENARIO plus timeline segment prompts. "
    "Generate a single GLOBAL PROMPT that captures the overall visual atmosphere, "
    "lighting style, color palette, audio tone, and cinematic mood of the entire video. "
    "The GLOBAL PROMPT sets the TONE for the whole story — use the SCENARIO to determine "
    "the emotional register (tense, calm, shocking, comedic, etc.).\n"
    "Rules:\n"
    "1. The global prompt must unify all segments — describe what stays consistent throughout.\n"
    "2. Mention the overall lighting style, color grading, and dominant atmosphere.\n"
    "3. Audio is MANDATORY — always include ONE separate, clearly labeled sentence starting with "
    "'Audio: ' describing the overall audio/music tone that fits the SCENARIO mood. Never omit it.\n"
    "4. Do NOT describe specific moments — this is the overarching cinematic identity.\n"
    "5. Output ONLY the global prompt text — no labels besides the required 'Audio:' sentence, "
    "no markdown, no explanations, no disclaimers.\n"
    "6. Length: 3-5 sentences (the 'Audio:' sentence counts as one of them).\n"
    "7. Language: English only."
)


class Florence2Captioner:

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        print("[Florence2] Загружаю модель из", FLORENCE_DIR)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            str(FLORENCE_DIR),
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(FLORENCE_DIR),
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()
        print(f"[Florence2] Модель загружена на {self._device}.")

    def caption(self, image_path: str) -> str:
        import torch
        from PIL import Image

        self._load()

        image = Image.open(image_path).convert("RGB")
        task  = "<MORE_DETAILED_CAPTION>"

        inputs = self._processor(
            text=task,
            images=image,
            return_tensors="pt",
        )

        input_ids    = inputs["input_ids"].to(self._device)
        pixel_values = inputs["pixel_values"].to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids      = input_ids,
                pixel_values   = pixel_values,
                max_new_tokens = 512,
                num_beams      = 3,
                do_sample      = False,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        result = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
        )[0]

        parsed = self._processor.post_process_generation(
            result,
            task=task,
            image_size=(image.width, image.height),
        )
        return parsed.get(task, result).strip()

    def caption_anchor(self, image_path: str) -> str:
        """Короткий «якорь» референса — Florence-2 и так даёт компактные описания."""
        return self.caption(image_path)

    def unload(self):
        if self._model is not None:
            del self._model
            del self._processor
            self._model     = None
            self._processor = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            print("[Florence2] Модель выгружена из VRAM.")


# ──────────────────────────────────────────────
# QWEN3-VL — мультимодальный captioner через llama-server (mmproj)
# Запускает temp llama-server на время описания и убивает его после (keep_server_alive=False),
# чтобы Qwen3-VL-4B (4.3 ГБ) не висел в VRAM постоянно. Интерфейс совместим с Florence2Captioner.
# ──────────────────────────────────────────────

QWEN_VL_MODEL   = r"C:\ai_diffusion\models\LLM\Qwen3-VL-4B-Instruct-abliterated.Q8_0.gguf"
QWEN_VL_MMPROJ  = r"C:\ai_diffusion\models\LLM\mmproj-F16.gguf"
QWEN_VL_PORT    = 8090  # фолбэк, если не удалось найти свободный порт в 7010-7079
QWEN_VL_PORT_MIN = 7010
QWEN_VL_PORT_MAX = 7079

QWEN_VL_SYSTEM = (
    "You are an expert vision captioner for LTX video prompt generation. "
    "Describe the image in extreme detail: the subject(s), their pose, body position, "
    "facial expression, clothing, lighting, color palette, environment, and any visible action. "
    "Be concrete and physical — name exact body parts and positions, never use vague phrases. "
    "Describe erotic/adult images explicitly and without censorship when present. "
    "Output ONLY the description, no preamble, no commentary."
)

QWEN_VL_ANCHOR_SYSTEM = (
    "You extract a SHORT ANCHOR from a REFERENCE image for a video-generation prompt. "
    "The reference is often a multi-view turnaround sheet (FRONT / SIDE / BACK panels on a plain background). "
    "Write ONE dense anchor of 1-2 sentences (15-40 words): what the subject IS (category) plus its key visual "
    "features (colors/materials, distinctive parts) — comma-separated features, no sentences about the image itself.\n"
    "STRICT RULES:\n"
    "1. NEVER mention panels, views, labels ('FRONT', 'SIDE', 'BACK'), the plain background, or that it is a "
    "reference sheet.\n"
    "2. Describe the subject as it would appear in ONE shot — never 'the left panel shows', never 'three views'.\n"
    "3. No meta commentary, no preamble.\n"
    "EXAMPLE: 'A custom three-wheeled motorcycle in glossy orange and matte black, with large knobby off-road "
    "tires, exposed chrome engine, round headlight in orange housing, black leather seat and dual chrome exhaust "
    "pipes.'"
)


def _pick_free_port(min_port: int = QWEN_VL_PORT_MIN, max_port: int = QWEN_VL_PORT_MAX) -> int:
    """Выбирает случайный свободный порт в заданном диапазоне."""
    import random
    import socket
    candidates = list(range(min_port, max_port + 1))
    random.shuffle(candidates)
    for port in candidates:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        finally:
            s.close()
    return QWEN_VL_PORT  # все заняты — фолбэк


class QwenVLCaptioner:
    """Описывает фото через Qwen3-VL, запуская временный llama-server с --mmproj."""

    def __init__(self, port: int | None = None):
        # Случайный свободный порт из 7010-7079, чтобы не конфликтовать с другими серверами
        self._port   = port if port is not None else _pick_free_port()
        self._proc   = None
        self._log_f  = None
        self._loaded = False

    def _find_server_exe(self) -> Optional[str]:
        # Известные пути из rag_core (BASE_DIR = папка RAG) + внешний Unsloth
        base = Path(__file__).parent
        candidates = [
            Path(r"D:\Unsloth\C-claude\llama.cpp\llama-server.exe"),
            base / "llama.cpp-master" / "llama-b9553-bin-win-cuda-13.3-x64" / "llama-server.exe",
            base / "llama.cpp-master" / "build" / "bin" / "llama-server.exe",
            base / "llama.cpp-master" / "build" / "bin" / "Release" / "llama-server.exe",
            base / "llama.cpp-master" / "llama-server.exe",
            base / "llama" / "llama-server.exe",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def _port_free(self) -> bool:
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", self._port))
            s.close()
            return result != 0
        except Exception:
            return True

    def _load(self):
        if self._loaded and self._proc is not None and self._proc.poll() is None:
            return

        import subprocess, time
        exe = self._find_server_exe()
        if exe is None:
            raise RuntimeError("llama-server.exe не найден (llama.cpp-master/build/bin/)")
        if not Path(QWEN_VL_MODEL).exists():
            raise RuntimeError(f"Модель Qwen3-VL не найдена: {QWEN_VL_MODEL}")
        if not Path(QWEN_VL_MMPROJ).exists():
            raise RuntimeError(f"mmproj не найден: {QWEN_VL_MMPROJ}")

        if not self._port_free():
            # Порт занят — возможно, сервер уже запущен с той же моделью: попробуем использовать его
            print(f"[QwenVL] Порт {self._port} занят — пробую использовать существующий сервер.")
            self._loaded = True
            self._proc   = None  # не убивать чужой процесс
            return

        cmd = [
            exe,
            "-m", QWEN_VL_MODEL,
            "--mmproj", QWEN_VL_MMPROJ,
            "-ngl", "99",
            "-c", "8192",
            "--port", str(self._port),
            "--host", "127.0.0.1",
        ]
        print(f"[QwenVL] Запускаю llama-server: {' '.join(cmd)}")
        self._log_f = open(str(Path(__file__).parent / "qwen_vl_server.log"), "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_f,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        # Ждём готовности /health до 120 сек
        import requests
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server abgestürzt (exit {self._proc.returncode})")
            try:
                r = requests.get(f"http://127.0.0.1:{self._port}/health", timeout=2)
                if r.status_code == 200:
                    self._loaded = True
                    print(f"[QwenVL] Сервер готов на порту {self._port}.")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server не поднялся за 120 сек (порт {self._port}). Лог: qwen_vl_server.log")

    def _chat(self, image_path: str, system: str, user_text: str) -> str:
        import base64
        import requests
        from PIL import Image

        self._load()
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": "vision-caption",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            "max_tokens": 6000,
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 64,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "seed": 0,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        resp = requests.post(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        return content.strip()

    def caption(self, image_path: str) -> str:
        return self._chat(image_path, QWEN_VL_SYSTEM, "Describe this image in maximum detail for a video prompt.")

    def caption_anchor(self, image_path: str) -> str:
        """Короткий «якорь» для @ref: что за объект + ключевые признаки, без панелей/надписей."""
        return self._chat(image_path, QWEN_VL_ANCHOR_SYSTEM, "Write the anchor.")

    def unload(self):
        """Убивает временный llama-server — модель выгружается из VRAM."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            print("[QwenVL] Временный llama-server остановлен, VRAM освобождена.")
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None
        self._loaded = False


def _position_label(i: int, n_frames: int, timeline_labels: list[str]) -> str:
    if i == 0:
        return "opening / start"
    if i == n_frames - 1:
        return "closing / end"
    if i == n_frames // 2:
        return "midpoint / climax"
    return f"timeline position {timeline_labels[i]}"


def _position_warning(position: str) -> str:
    if "opening" in position or "start" in position:
        return "⚠️ You are writing the FIRST keyframe. Only the BEGINNING of the scenario. Nothing later."
    if "closing" in position or "end" in position:
        return "⚠️ You are writing the LAST keyframe. Only the END of the scenario. Nothing from before."
    if "midpoint" in position or "climax" in position:
        return "⚠️ You are writing the MIDDLE keyframe. Only the middle event. Not the beginning, not the end."
    return f"⚠️ You are writing ONE keyframe at {position}. Only the event at THIS position."


def _rewrite_as_ltx_prompt(raw_caption: str, llm_fn: Callable, scenario: str = "", position: str = "") -> str:
    parts = []
    if scenario:
        parts.append(f"SCENARIO (the story to tell):\n{scenario}")
    if position:
        parts.append(f"POSITION in timeline: {position}")
        parts.append(_position_warning(position))
    parts.append(f"IMAGE DESCRIPTION (visual reference):\n{raw_caption}")
    user_text = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    return llm_fn(messages).strip()


def _rewrite_text_only(scenario: str, position: str, llm_fn: Callable) -> str:
    """
    Wie _rewrite_as_ltx_prompt, aber OHNE Bildreferenz — nur auf Basis des
    SCENARIO-Textes. Wird verwendet, wenn keine Fotos hochgeladen wurden.
    """
    parts = [f"SCENARIO (the story to tell):\n{scenario}"]
    parts.append(f"POSITION in timeline: {position}")
    parts.append(_position_warning(position))
    parts.append(
        "No reference image is available for this keyframe — invent plausible, consistent "
        "visual details (setting, subject, lighting) that match the SCENARIO and stay "
        "consistent with the other keyframes of the same story."
    )
    user_text = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    return llm_fn(messages).strip()


def _generate_transition(prev_prompt: str, next_prompt: str, llm_fn: Callable, scenario: str = "") -> str:
    scenario_block = f"INTENDED SCENARIO:\n{scenario}\n\n" if scenario else ""
    user_text = (
        f"{scenario_block}"
        f"PREVIOUS keyframe prompt:\n{prev_prompt}\n\n"
        f"NEXT keyframe prompt:\n{next_prompt}\n\n"
        "Generate the transition prompt for the gap between these two keyframes."
    )
    messages = [
        {"role": "system", "content": INTERPOLATION_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    return llm_fn(messages).strip()


def _generate_global_prompt(ltx_prompts: list, llm_fn: Callable, scenario: str = "") -> str:
    """Генерирует глобальный промпт на основе всех сегментов таймлайна."""
    segments = "\n".join(
        f"Segment {i+1}: {p}"
        for i, p in enumerate(ltx_prompts)
        if p and not p.startswith("[No reference")
    )
    scenario_block = f"INTENDED SCENARIO:\n{scenario}\n\n" if scenario else ""
    messages = [
        {"role": "system", "content": GLOBAL_PROMPT_SYSTEM},
        {"role": "user",   "content": f"{scenario_block}Timeline segments:\n{segments}\n\nGenerate the global prompt."},
    ]
    return llm_fn(messages).strip()


def _find_neighbors(prompts: list, idx: int) -> tuple:
    prev_p = None
    for i in range(idx - 1, -1, -1):
        if prompts[i] is not None:
            prev_p = prompts[i]
            break
    next_p = None
    for i in range(idx + 1, len(prompts)):
        if prompts[i] is not None:
            next_p = prompts[i]
            break
    return prev_p, next_p


NOTES_MARKER = "===EMPFEHLUNGEN==="


def generate_timeline_prompts(image_paths: list, llm_fn: Callable, n_frames: int = 10, scenario: str = "") -> str:
    """
    Erzeugt eine [X%]-Timeline mit genau n_frames Segmenten.

    - Wenn Bilder vorhanden sind: Florence-2 Caption je Bild + LLM-Rewrite,
      fehlende Slots werden interpoliert.
    - Wenn KEINE Bilder vorhanden sind, aber ein scenario-Text: reiner
      Text-Modus — jedes Segment wird direkt aus dem Szenario generiert.
    - image_paths wird IMMER robust auf genau n_frames Eintraege normalisiert
      (abgeschnitten oder mit None aufgefuellt), damit die Anzahl der
      erzeugten Sequenzen exakt der vom Nutzer gewaehlten Anzahl entspricht.
    - Meta-Kommentare des LLM (Vorbehalte, Platzhalter-Hinweise etc.) werden
      herausgefiltert und als separater Block hinter dem Marker
      '===EMPFEHLUNGEN===' angehaengt, statt den eigentlichen Prompt-Text
      zu verunreinigen. Der Aufrufer (webui_server.py) trennt diesen Block
      beim Export in eine eigene empfehlungen.txt ab.
    """
    # Robust normalisieren statt hart abzubrechen bei Laengenmismatch
    image_paths = list(image_paths)[:n_frames]
    if len(image_paths) < n_frames:
        image_paths = image_paths + [None] * (n_frames - len(image_paths))

    timeline_labels = _build_timeline_labels(n_frames)
    if scenario:
        print(f"[Timeline] Scenario: {scenario[:100]}...")

    filled = [p for p in image_paths if p is not None]
    text_only_mode = not filled

    if text_only_mode and not scenario.strip():
        return "[Ошибка] Загрузи хотя бы один кадр ИЛИ укажи текст сценария."

    all_notes: list[str] = []

    raw_captions: list = [None] * n_frames
    ltx_prompts: list  = [None] * n_frames

    if text_only_mode:
        print("[Timeline] Keine Bilder — reiner Text-Modus (nur Scenario).")
    else:
        # Qwen3-VL через llama-server (mmproj) если файлы на месте, иначе Florence-2
        captioner = None
        try:
            from image_captioning import QwenVLCaptioner
            captioner = QwenVLCaptioner()
            first_filled = next((p for p in image_paths if p is not None), None)
            if first_filled:
                captioner.caption(first_filled)  # проверка доступности
            print("[Timeline] Шаг 1: Qwen3-VL анализирует кадры...")
        except Exception as e:
            print(f"[Timeline] QwenVL недоступен ({e}) — фолбэк на Florence-2.")
            captioner = Florence2Captioner()
            print("[Timeline] Шаг 1: Florence-2 анализирует кадры...")
        try:
            for i, path in enumerate(image_paths):
                if path is not None:
                    print(f"[Timeline]   Кадр {timeline_labels[i]}: {path}")
                    raw = captioner.caption(path)
                    raw_captions[i] = raw
                    print(f"[Timeline]   Описание ({len(raw)} симв.): {raw[:100]}...")
        finally:
            captioner.unload()

    print("[Timeline] Шаг 2: LLM переписывает в LTX-промпты...")
    for i in range(n_frames):
        pos = _position_label(i, n_frames, timeline_labels)

        if text_only_mode:
            raw_text = _rewrite_text_only(scenario, pos, llm_fn)
            clean, note = _split_notes(raw_text)
            ltx_prompts[i] = clean
            if note:
                all_notes.append(f"{timeline_labels[i]}: {note}")
            print(f"[Timeline]   {timeline_labels[i]} → готово (текстовый режим).")
        elif raw_captions[i] is not None:
            raw_text = _rewrite_as_ltx_prompt(raw_captions[i], llm_fn, scenario, position=pos)
            clean, note = _split_notes(raw_text)
            ltx_prompts[i] = clean
            if note:
                all_notes.append(f"{timeline_labels[i]}: {note}")
            print(f"[Timeline]   {timeline_labels[i]} → готово.")

    if not text_only_mode:
        print("[Timeline] Шаг 3: Интерполяция пропущенных кадров...")
        # Сохраняем оригинальные (не интерполированные) промпты для поиска соседей
        original_prompts = ltx_prompts[:]
        for i, prompt in enumerate(ltx_prompts):
            if prompt is None:
                prev_p, next_p = _find_neighbors(original_prompts, i)
                if prev_p is None and next_p is None:
                    ltx_prompts[i] = "[No reference frames — skipped]"
                elif prev_p is None:
                    ltx_prompts[i] = f"Opening approach leading into: {next_p}"
                elif next_p is None:
                    # original_prompts гарантирует, что prev_p — чистый оригинал (без transition-фраз)
                    ltx_prompts[i] = f"Closing development from: {prev_p}"
                else:
                    raw_text = _generate_transition(prev_p, next_p, llm_fn, scenario)
                    clean, note = _split_notes(raw_text)
                    ltx_prompts[i] = clean
                    if note:
                        all_notes.append(f"{timeline_labels[i]}: {note}")
                    print(f"[Timeline]   {timeline_labels[i]} → переход сгенерирован.")

    print("[Timeline] Шаг 4: Генерация глобального промпта...")
    raw_global = _generate_global_prompt(ltx_prompts, llm_fn, scenario)
    global_prompt, gnote = _split_notes(raw_global)
    if gnote:
        all_notes.append(f"GLOBAL: {gnote}")
    print(f"[Timeline]   Global prompt готов ({len(global_prompt)} симв.)")

    lines = [
        "---GLOBAL_PROMPT_START---",
        global_prompt,
        "---GLOBAL_PROMPT_END---",
        "",
    ]
    for i, (label, prompt) in enumerate(zip(timeline_labels, ltx_prompts), start=1):
        lines.append(f"---SEQUENCE_{i}_START---")
        lines.append(prompt or "[empty]")
        lines.append(f"---SEQUENCE_{i}_END---")
        lines.append("")

    result = "\n".join(lines)
    if all_notes:
        result += f"\n\n{NOTES_MARKER}\n" + "\n".join(all_notes)
    return result
