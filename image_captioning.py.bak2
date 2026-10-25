"""
image_captioning.py — Florence-2 мульти-кадровый генератор промптов для LTX 2.3.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import gc

FLORENCE_DIR = Path(__file__).parent / "florence-2-large"
TIMELINE_LABELS = ["[0%]", "[20%]", "[40%]", "[60%]", "[80%]"]

REWRITE_SYSTEM = (
    "You are an expert LTX Video 2.3 prompt engineer. "
    "Convert a raw image description into a precise, cinematic LTX Director prompt. "
    "Rules:\n"
    "1. Start with camera movement/position (e.g. 'Static wide shot', 'Slow push-in', 'Low-angle tracking').\n"
    "2. Describe the subject, action, and environment with rich visual detail.\n"
    "3. Include lighting quality, color grading, and atmosphere.\n"
    "4. End with audio texture (ambient sound, music tone, or silence).\n"
    "5. Output ONLY the prompt text — no labels, no markdown, no explanations.\n"
    "6. Length: 2-4 sentences maximum.\n"
    "7. Language: English only."
)

INTERPOLATION_SYSTEM = (
    "You are an expert LTX Video 2.3 prompt engineer specializing in scene transitions. "
    "You receive the PREVIOUS keyframe prompt and the NEXT keyframe prompt. "
    "Generate a smooth cinematic transition prompt for the gap between them. "
    "Rules:\n"
    "1. Logically connect both scenes — camera movement, action development, or environment shift.\n"
    "2. Start with camera movement (e.g. 'Slow pan reveals', 'Camera drifts from X toward Y').\n"
    "3. Describe what changes: subject movement, lighting shift, or spatial transition.\n"
    "4. Keep the same visual atmosphere and color palette as neighboring frames.\n"
    "5. Output ONLY the transition prompt — no labels, no markdown, no explanations.\n"
    "6. Length: 2-4 sentences maximum.\n"
    "7. Language: English only."
)

GLOBAL_PROMPT_SYSTEM = (
    "You are an expert LTX Video 2.3 prompt engineer. "
    "You receive a series of timeline segment prompts for a video. "
    "Generate a single GLOBAL PROMPT that captures the overall visual atmosphere, "
    "lighting style, color palette, audio tone, and cinematic mood of the entire video. "
    "Rules:\n"
    "1. The global prompt must unify all segments — describe what stays consistent throughout.\n"
    "2. Mention the overall lighting style, color grading, and dominant atmosphere.\n"
    "3. Include the overall audio/music tone (ambient, tense, upbeat, silent, etc.).\n"
    "4. Do NOT describe specific moments — this is the overarching cinematic identity.\n"
    "5. Output ONLY the global prompt text — no labels, no markdown, no explanations.\n"
    "6. Length: 3-5 sentences.\n"
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
                input_ids    = input_ids,
                pixel_values = pixel_values,
                max_new_tokens = 512,
                num_beams    = 3,
                do_sample    = False,
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


def _rewrite_as_ltx_prompt(raw_caption: str, llm_fn: Callable) -> str:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user",   "content": f"Raw image description:\n{raw_caption}"},
    ]
    return llm_fn(messages).strip()


def _generate_transition(prev_prompt: str, next_prompt: str, llm_fn: Callable) -> str:
    user_text = (
        f"PREVIOUS keyframe prompt:\n{prev_prompt}\n\n"
        f"NEXT keyframe prompt:\n{next_prompt}\n\n"
        "Generate the transition prompt for the gap between these two keyframes."
    )
    messages = [
        {"role": "system", "content": INTERPOLATION_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    return llm_fn(messages).strip()


def _generate_global_prompt(ltx_prompts: list, llm_fn: Callable) -> str:
    """Генерирует глобальный промпт на основе всех сегментов таймлайна."""
    segments = "\n".join(
        f"Segment {i+1}: {p}"
        for i, p in enumerate(ltx_prompts)
        if p and not p.startswith("[No reference")
    )
    messages = [
        {"role": "system", "content": GLOBAL_PROMPT_SYSTEM},
        {"role": "user",   "content": f"Timeline segments:\n{segments}\n\nGenerate the global prompt."},
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


def generate_timeline_prompts(image_paths: list, llm_fn: Callable) -> str:
    if len(image_paths) != 5:
        return "[Ошибка] Нужно ровно 5 путей (None для пропущенных кадров)."

    filled = [p for p in image_paths if p is not None]
    if not filled:
        return "[Ошибка] Загрузи хотя бы один кадр."

    captioner   = Florence2Captioner()
    raw_captions: list = [None] * 5
    ltx_prompts: list  = [None] * 5

    try:
        print("[Timeline] Шаг 1: Florence-2 анализирует кадры...")
        for i, path in enumerate(image_paths):
            if path is not None:
                print(f"[Timeline]   Кадр {TIMELINE_LABELS[i]}: {path}")
                raw = captioner.caption(path)
                raw_captions[i] = raw
                print(f"[Timeline]   Описание ({len(raw)} симв.): {raw[:100]}...")
    finally:
        captioner.unload()

    print("[Timeline] Шаг 2: LLM переписывает в LTX-промпты...")
    for i, raw in enumerate(raw_captions):
        if raw is not None:
            ltx_prompts[i] = _rewrite_as_ltx_prompt(raw, llm_fn)
            print(f"[Timeline]   {TIMELINE_LABELS[i]} → готово.")

    print("[Timeline] Шаг 3: Интерполяция пропущенных кадров...")
    for i, prompt in enumerate(ltx_prompts):
        if prompt is None:
            prev_p, next_p = _find_neighbors(ltx_prompts, i)
            if prev_p is None and next_p is None:
                ltx_prompts[i] = "[No reference frames — skipped]"
            elif prev_p is None:
                ltx_prompts[i] = f"Opening approach leading into: {next_p}"
            elif next_p is None:
                ltx_prompts[i] = f"Continuation and fade from: {prev_p}"
            else:
                ltx_prompts[i] = _generate_transition(prev_p, next_p, llm_fn)
                print(f"[Timeline]   {TIMELINE_LABELS[i]} → переход сгенерирован.")

    print("[Timeline] Шаг 4: Генерация глобального промпта...")
    global_prompt = _generate_global_prompt(ltx_prompts, llm_fn)
    print(f"[Timeline]   Global prompt готов ({len(global_prompt)} симв.)")

    lines = [
        "═" * 60,
        "🎬  LTX 2.3 MULTI-FRAME TIMELINE PROMPTS",
        "═" * 60,
        "",
        "🌐 GLOBAL PROMPT",
        "─" * 60,
        global_prompt,
        "",
        "═" * 60,
        "📍 TIMELINE SEGMENTS",
        "═" * 60,
        "",
    ]
    for i, (label, prompt) in enumerate(zip(TIMELINE_LABELS, ltx_prompts)):
        source = "📷 Image" if image_paths[i] is not None else "✨ Generated"
        lines.append(f"{label}  [{source}]")
        lines.append(prompt or "[empty]")
        lines.append("")

    lines.append("─" * 60)
    lines.append("💡 TIP: Copy GLOBAL PROMPT into the LTX 2.3 global node, then each segment into its Director node.")
    return "\n".join(lines)
