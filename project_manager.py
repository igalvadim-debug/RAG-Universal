"""
project_manager.py — управление проектами и генерация FLUX T2I промптов.
Создаёт структуру папок projects/scenario_NAME_NN/ с промптами для каждой сцены.
"""

import re
import json
from pathlib import Path
from datetime import datetime

import rag_core

BASE_DIR     = Path(__file__).parent
PROJECTS_DIR = BASE_DIR / "projects"

# ──────────────────────────────────────────────
# МАППИНГ: Narrator → FLUX-стиль
# ──────────────────────────────────────────────

FLUX_STYLE_MAP = {
    # Cinematic
    "короткий фильм":           "cinematic photorealistic, 35mm film, soft bokeh, natural dramatic lighting, anamorphic lens flare, shallow depth of field, color graded",
    "боевик":                   "cinematic action shot, dynamic composition, high contrast lighting, dramatic shadows, photorealistic, 24mm wide lens, motion blur",
    "детектив":                 "film noir cinematic, high contrast black and shadow, rain-slicked streets, venetian blind shadows, dramatic chiaroscuro lighting, photorealistic",
    "рассказ девушки":          "cinematic portrait, soft natural light, shallow depth of field, photorealistic, warm color grading, 50mm lens, emotional expression",
    "рассказ парня":            "cinematic portrait, dramatic side lighting, photorealistic, 50mm lens, gritty realistic atmosphere, shallow depth of field",
    "рассказ старушки":         "cinematic portrait, soft window light, photorealistic, elderly woman, warm nostalgic color grading, 85mm portrait lens",
    "рассказ старика":          "cinematic portrait, dramatic rim lighting, photorealistic, elderly man, desaturated color grading, 85mm portrait lens",
    # Horror
    "хоррор":                   "horror atmospheric, dark and menacing, deep shadows, eerie cold blue lighting, fog, unsettling composition, ultra photorealistic, high contrast",
    # Anime / Illustration
    "скетч":                    "sketch illustration style, ink linework, light watercolor wash, loose expressive strokes, black and white with subtle color accents",
    "комедия":                  "bright colorful cartoon illustration, clean lines, vibrant saturated colors, comic book style, expressive character design",
    # Cartoon / Animation / Silent film
    "немое кино":               "black and white vintage 1920s illustration, grainy film texture, Art Deco style, expressionist shadows, halftone print effect",
    "немое кино (фортепиано)":  "romantic black and white vintage illustration, soft pencil sketch style, 1920s atmosphere, gentle shadows, nostalgic mood",
    # Documentary
    "документация":             "documentary photography style, natural ambient light, photojournalism, candid composition, neutral color palette, sharp focus, Leica reportage",
    # Dark Fantasy
    "мрачный":                  "dark fantasy concept art, dramatic atmospheric lighting, deep shadows, gothic mood, painterly style, rich dark color palette, hyper-detailed",
    # Sci-Fi
    "sci-fi":                   "science fiction concept art, futuristic environment, neon holographic lighting, sleek metallic surfaces, cold blue tones, hyper-detailed CGI render",
    # Talkshow
    "Talkshow":                 "bright studio photography, three-point lighting, clean professional background, photorealistic, sharp focus, broadcast quality",
    # Music / Concert
    "music: rock concert":      "rock concert photography, dramatic stage lighting, smoke and spotlights, dynamic composition, high contrast, wide angle, photorealistic",
    "music: pop concert":       "pop concert photography, colorful LED stage lights, vibrant atmosphere, confetti, wide angle, photorealistic, energetic crowd",
    "music: русский шансон":    "gritty post-Soviet photography, warm tungsten light, cigarette smoke, intimate atmosphere, desaturated tones, photorealistic documentary",
    "music: latino pop":        "vibrant sunny photography, golden hour light, tropical colors, warm saturated tones, photorealistic, festive atmosphere",
    "music: latin reggaeton":   "urban night photography, neon lights, high contrast, photorealistic, dynamic low-angle composition, vibrant saturated colors",
}

FLUX_STYLE_DEFAULT = "cinematic photorealistic, natural dramatic lighting, shallow depth of field, color graded, high detail"


def get_flux_style(narrator: str) -> str:
    """Возвращает FLUX-стиль суффикс для заданного нарратора."""
    return FLUX_STYLE_MAP.get(narrator, FLUX_STYLE_DEFAULT)


def _slugify(text: str, max_len: int = 30) -> str:
    """Превращает произвольный текст в безопасное имя папки."""
    text = text.lower().strip()
    translit = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh',
        'з':'z','и':'i','й':'y','к':'k','л':'l','м':'m','н':'n','о':'o',
        'п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts',
        'ч':'ch','ш':'sh','щ':'shch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
    }
    result = ""
    for ch in text:
        result += translit.get(ch, ch)
    result = re.sub(r'[^a-z0-9]+', '_', result)
    result = result.strip('_')
    return result[:max_len] if result else "scenario"


def _next_project_index(slug: str) -> int:
    """Находит следующий свободный номер для папки с данным slug."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]
    idx = 1
    while f"scenario_{slug}_{idx:02d}" in existing:
        idx += 1
    return idx


def _calc_pcts(n: int) -> list[str]:
    """Рассчитывает равномерные проценты таймлайна для n сцен."""
    if n == 1:
        return ["0%"]
    step = 100 // (n - 1)
    pcts = [f"{i * step}%" for i in range(n)]
    pcts[-1] = "100%"
    return pcts


# ──────────────────────────────────────────────
# QUALITY BOILERPLATE — anhängen an jeden Prompt
# ──────────────────────────────────────────────
QUALITY_SUFFIX = (
    "masterwork, masterpiece, best quality, ultra HD, 8k resolution, "
    "hyper-detailed, razor-sharp, professional lighting, HDR, "
    "taken with Canon EOS R5, DSLR, 75mm lens, depth of field, cinematic"
)


def generate_flux_prompts_llm(
    question:       str,
    narrator:       str,
    n_sequences:    int,
    cloud_provider: str = "auto",
) -> list[str]:
    """
    Генерирует через LLM список ausführlicher FLUX T2I-Prompts
    im cinematischen Prosa-Stil — extrem detailliert, ähnlich einer
    Drehbuch-Kamerabeschreibung mit technischen Qualitäts-Terms.
    """
    flux_style = get_flux_style(narrator)
    pcts       = _calc_pcts(n_sequences)
    pct_list   = ", ".join(pcts)

    system_prompt = (
        "You are a world-class cinematographer and FLUX prompt engineer. "
        "Your task: generate extremely detailed, cinematic prose image prompts for a video storyboard. "
        "Each prompt must read like a director of photography describing a single frozen frame — "
        "rich, sensory, hyper-detailed, using the language of film and photography.\n\n"
        "CRITICAL RULES:\n"
        "1. Write in flowing, vivid English prose — NOT bullet points, NOT comma-separated tags, NOT parenthesized lists.\n"
        "2. Describe the frame as if you're narrating it: composition first, then lighting, then textures, then mood.\n"
        "3. Be OBSESSIVELY detailed about: lighting direction & quality, color temperature, textures, materials, "
        "skin details, fabric weaves, reflections, shadows, depth of field, bokeh, lens characteristics.\n"
        "4. Mention the camera/lens perspective: close-up / macro / medium shot / wide shot, angle (from above / below / Dutch), focal length feel.\n"
        "5. Every prompt MUST end with this EXACT suffix (append verbatim, no changes):\n"
        f"   {QUALITY_SUFFIX}\n"
        "6. Length: 100–180 words of vivid description BEFORE the suffix.\n"
        "7. ONE static frame per prompt — no video, no action sequences, no dialogue, no story progression within a single prompt.\n"
        "8. Output ONLY a valid JSON array of strings, NO markdown, NO code fences, NO explanation:\n"
        '   ["prompt 1", "prompt 2", ...]\n\n'
        "EXAMPLE of ONE correct prompt (study the style, detail density, and rhythm):\n"
        '"a razor-sharp macro close-up of a mesmerizing human eye, capturing every intricate detail with ultra HD precision. '
        'The golden brown iris dominates the right side of the frame, its fibrous texture rendered in unflinching clarity, '
        'each delicate strand catching dappled light that makes the eye seem almost alive. The pupil is ink-dark and fully dilated, '
        'surrounded by a dewy corneal surface that reflects prismatic highlights, emphasizing the glossy sheen. '
        'Thick, voluminous lashes frame the eye, arching elegantly with a hint of cinematic movement, while the lower lashes remain defined. '
        'On the left side, a metallic gold rectangle floats like a luxurious badge, its sleek branding radiating a subtle glow '
        'against the void-like black background, which deepens the electric tones of the iris. '
        'The composition balances hyperreal drama with high-tech aesthetics, making it an obsession for next-gen resolution. '
        f'{QUALITY_SUFFIX}"'
    )

    user_prompt = (
        f"SCENE CONCEPT / STORY: {question}\n"
        f"VISUAL STYLE DIRECTION: {flux_style}\n\n"
        f"Generate exactly {n_sequences} FLUX image prompts for these storyboard positions: {pct_list}\n"
        f"Story arc: position {pcts[0]} establishes setting/character → position {pcts[-1]} delivers climax/resolution.\n"
        f"Incorporate the visual style organically into the prose (mood, lighting, color palette, atmosphere).\n\n"
        f"Return ONLY a JSON array with exactly {n_sequences} strings. "
        f"Each string is ONE complete cinematic prose prompt ending with the mandatory quality suffix."
    )

    raw = rag_core.call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=3500,
        temperature=0.75,
        provider=cloud_provider,
    )

    # Парсим JSON из ответа
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            prompts = json.loads(match.group(0))
            if isinstance(prompts, list) and len(prompts) == n_sequences:
                return [str(p) for p in prompts]
    except Exception:
        pass

    # Фолбэк: разбиваем по строкам
    lines = [
        l.strip().strip('"').strip("',")
        for l in raw.splitlines()
        if l.strip() and not l.strip().startswith('[') and not l.strip().startswith(']')
    ]
    lines = [l for l in lines if len(l) > 150]
    if len(lines) >= n_sequences:
        return lines[:n_sequences]

    while len(lines) < n_sequences:
        lines.append(f"[Scene {len(lines)+1}] — prompt generation failed, please retry")
    return lines


def create_project(
    question:       str,
    narrator:       str,
    n_sequences:    int,
    cloud_provider: str = "auto",
) -> tuple[str, str]:
    """
    Создаёт папку проекта, генерирует FLUX-промпты, сохраняет scene_N.txt.

    Возвращает:
        (project_path_str, display_text)
    """
    if not question.strip():
        return "", "⚠️ Введи тему/идею в поле запроса перед созданием проекта."

    slug      = _slugify(question, max_len=25)
    idx       = _next_project_index(slug)
    proj_name = f"scenario_{slug}_{idx:02d}"
    proj_dir  = PROJECTS_DIR / proj_name
    proj_dir.mkdir(parents=True, exist_ok=True)

    pcts       = _calc_pcts(n_sequences)
    flux_style = get_flux_style(narrator)

    prompts = generate_flux_prompts_llm(question, narrator, n_sequences, cloud_provider)

    for i, prompt in enumerate(prompts, start=1):
        scene_file = proj_dir / f"scene_{i}.txt"
        scene_file.write_text(prompt, encoding="utf-8")

    meta = {
        "created":        datetime.now().isoformat(),
        "question":       question,
        "narrator":       narrator,
        "n_sequences":    n_sequences,
        "flux_style":     flux_style,
        "cloud_provider": cloud_provider,
        "timeline_pcts":  pcts,
    }
    (proj_dir / "project_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"📁 Проект создан: {proj_dir}",
        f"🎬 Нарратор: {narrator}",
        f"🎨 FLUX-стиль: {flux_style}",
        f"📸 Сцен: {n_sequences}  ({' | '.join(pcts)})",
        "",
        "═" * 60,
    ]
    for i, (pct, prompt) in enumerate(zip(pcts, prompts), start=1):
        lines.append(f"\n🖼️  SCENE {i}  [{pct}]  →  scene_{i}.txt")
        lines.append("─" * 50)
        lines.append(prompt)

    lines.append("\n" + "═" * 60)
    lines.append("✅ Промпты сохранены. Генерируй изображения в FLUX, загружай в слоты и запускай LTX 2.3!")

    return str(proj_dir), "\n".join(lines)
