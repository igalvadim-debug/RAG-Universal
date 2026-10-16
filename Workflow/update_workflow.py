import json
import os
import re
import glob
import time
from pathlib import Path
from datetime import datetime

# ==========================================
# KONFIGURATION
# ==========================================

SCRIPT_DIR     = Path(__file__).parent
USED_FILE      = SCRIPT_DIR / "used.md"
OUTPUT_JSON    = "modified_LTX_Dir.json"
JSON_PAIRS     = [
    ("low.json", "mod_low.json"),
    ("high.json", "mod_high.json"),
    # LTX 2.5 Director (нода LTXDirectorCS25) — принцип CS LTX Director
    ("LTX_2_5_Director_CStest.json", "mod_ltx25_director.json"),
]
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

# ------------------------------------------
# LTX Director 2.0 — widgets_values Layout
# ------------------------------------------
# 0  start_second
# 1  end_second
# 2  duration_seconds
# 3  start_frame
# 4  end_frame
# 5  duration_frames
# 6  timeline_data   (JSON-String, enthaelt "segments", "global_prompt", ...)
# 7  local_prompts   (Text, Segmente getrennt mit " | ")
# 8  segment_lengths (CSV-String, z.B. "53,66,55,37,59")
# 9  epsilon
# 10 guide_strength
# 11..22  diverse Flags/Settings (werden nicht angefasst)
W_TIMELINE_DATA = 6
W_LOCAL_PROMPTS = 7
W_SEGMENT_LENGTHS = 8


# ==========================================
# USED.MD — Buchfuehrung ueber verarbeitete Ordner
# ==========================================

def load_used() -> set:
    """Liest used.md und gibt Set mit bereits verwendeten Ordnernamen zurueck."""
    if not USED_FILE.exists():
        return set()
    used = set()
    for line in USED_FILE.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            used.add(name)
    return used


def mark_used(folder_name: str):
    """Haengt Ordnername an used.md an (mit Datum)."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(USED_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{folder_name}  # {stamp}\n")
    print(f"📝 '{folder_name}' in used.md vermerkt.")


# ==========================================
# ORDNER-SCOUTING
# ==========================================

def find_work_folder() -> Path | None:
    """
    Findet den naechsten unbearbeiteten Unterordner in SCRIPT_DIR.
    Ignoriert __pycache__.
    Sortiert nach Modifikationszeit (neueste zuerst).
    Bevorzugt Ordner, die NICHT in used.md stehen.
    """
    used = load_used()
    candidates = []

    for entry in SCRIPT_DIR.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "__pycache__":
            continue
        candidates.append(entry)

    if not candidates:
        return None

    # Neueste zuerst
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Zuerst unbenutzte, dann alle
    unused = [c for c in candidates if c.name not in used]
    if unused:
        print(f"📁 {len(unused)} unbenutzte Ordner gefunden.")
        return unused[0]

    # Alle schon benutzt — nimm den neuesten trotzdem
    print(f"⚠️  Alle {len(candidates)} Ordner bereits in used.md. Verwende neuesten.")
    return candidates[0]


# ==========================================
# PARSING
# ==========================================

def find_images(work_dir: Path) -> list[str]:
    """
    Sucht alle Bilder im work_dir, sortiert alphabetisch.
    Gibt nur Dateinamen zurueck.
    """
    images = set()
    for ext in IMG_EXTENSIONS:
        for f in work_dir.glob(f"*{ext}"):
            images.add(f.name)
        for f in work_dir.glob(f"*{ext.upper()}"):
            images.add(f.name)
    return sorted(images)


def _strip_meta(prompts: list) -> tuple:
    """Вырезает мета-теги LLM ([DURATION: X.Xs] и строки DURATION_WEIGHT: N 1-5)
    из текста промтов; возвращает (промты, weights)."""
    cleaned = []
    weights = []
    for p in prompts:
        w = None
        keep = []
        for ln in p.splitlines():
            ls = ln.strip()
            m = re.match(r"(?i)^DURATION\s*_?\s*WEIGHT\s*:\s*([1-5])\s*$", ls)
            if m:
                w = int(m.group(1))
                continue
            m2 = re.match(r"(?i)^\[\s*DURATION\s*:\s*[^\]]*\]\s*$", ls)
            if m2:
                continue
            keep.append(ln)
        cleaned.append("\n".join(keep).strip())
        weights.append(w)
    return cleaned, weights


def parse_prompt_file(filepath: Path) -> tuple[str, list[str], list]:
    """Liest video.txt / novideo.txt, extrahiert Global Prompt + Sequenzen + weights."""
    if not filepath.exists():
        print(f"❌ '{filepath.name}' nicht gefunden in {filepath.parent}")
        return "", []

    text = filepath.read_text(encoding="utf-8")
    global_prompt = ""
    prompts = []

    # ── CLEANUP: игнорируем всё ДО первого GLOBAL_PROMPT_START и ПОСЛЕ последнего SEQUENCE...END ──
    # Модель часто пишет мусор (нарратив, мета-комментарии) до глобального промта и после последней сцены.
    # Режем детерминированно, чтобы парсер видел только полезную часть.
    _gp_start = re.search(
        r"(?i)GLOBAL[\\\s_]*PROMPT[\\\s_]*START",
        text,
    )
    if _gp_start:
        text = text[_gp_start.start():]

    _end_markers = re.findall(
        r"(?i)SEQUEN[\\\s_]*[CT][\\\s_]*[A-Z]*[\\\s_]*\d+[\\\s_]*END",
        text,
    )
    if _end_markers:
        _last_end = max(text.lower().rfind(m.lower()) for m in _end_markers)
        _tail = text[_last_end:]
        _m = re.match(
            r"(?is)SEQUEN[\\\s_]*[CT][\\\s_]*[A-Z]*[\\\s_]*\d+[\\\s_]*END"
            r"[\\\s\\n]*(?:-{3})?[\\\s\\n]*",
            _tail,
        )
        if _m:
            text = text[:_last_end + _m.end()]

    # ── Format 1: machine delimiters ---GLOBAL_PROMPT_START--- / ---SEQUENCE_N_START---
    # Толерантно: допускает ---SEQUENCE_1_START---, "SEQUENCE 1 START", SEQUENCE_1_START,
    # и экранированные \_ (модель часто пишет SEQUENCE 1\_START) — любые комбинации разделителей.
    gp_match = re.search(
        r"(?:-{3}\s*)?GLOBAL[\\\s_]*PROMPT[\\\s_]*START\s*(?:-{3})?\s*(.*?)\s*(?:-{3}\s*)?GLOBAL[\\\s_]*PROMPT[\\\s_]*END\s*(?:-{3})?",
        text, re.DOTALL | re.IGNORECASE
    )
    if gp_match:
        global_prompt = re.sub(r'\s+', ' ', gp_match.group(1).strip())

    seq_matches = re.findall(
        r"(?:-{3}\s*)?SEQUEN[\\\s_]*[CT][\\\s_]*[A-Z]*[\\\s_]*\d+[\\\s_]*START\s*(?:-{3})?\s*(.*?)\s*(?:-{3}\s*)?SEQUEN[\\\s_]*[CT][\\\s_]*[A-Z]*[\\\s_]*\d+[\\\s_]*END\s*(?:-{3})?",
        text, re.DOTALL | re.IGNORECASE
    )
    if seq_matches:
        for m in seq_matches:
            # Preserve multi-line structure
            block = m.strip()
            block = re.sub(r'\n\s*\n', '\n', block)
            lines_list = [l.strip() for l in block.splitlines()]
            prompt_text = "\n".join(lines_list)
            if prompt_text:
                prompts.append(prompt_text)
        if prompts:
            print(f"  -> Format: SEQUENCE delimiters ({len(prompts)} sequences)")
            return global_prompt, *_strip_meta(prompts)

    # ── Format 2: Legacy GLOBAL PROMPT: + SEQUENCE X:
    global_match = re.search(
        r"(?:GLOBAL PROMPT|🌐 GLOBAL PROMPT)[:\-─\s]+(.*?)(?=\n\s*(?:─|═|SEQUENCE \d+:|\[\d+%\]|IMAGE \[|$))",
        text, re.DOTALL | re.IGNORECASE
    )
    if global_match:
        global_prompt = re.sub(r'\s+', ' ', global_match.group(1).strip())

    if "SEQUENCE" in text:
        blocks = text.split("SEQUENCE")
        for block in blocks:
            match = re.search(
                r"(?:Director Node|Cinematographer Directive):\s*(.*?)(?=\n\s*(?:Cinematographer Directive:|Audio Trigger:|Dialogue Trigger:|SEQUENCE \d+:|$))",
                block, re.DOTALL | re.IGNORECASE
            )
            if match:
                prompt_text = re.sub(r'\s+', ' ', match.group(1).strip())
                prompts.append(prompt_text)
        if prompts:
            print(f"  -> Format: legacy SEQUENCE ({len(prompts)} sequences)")
            return global_prompt, *_strip_meta(prompts)

    # ── Format 3a: Timeline [X%]: (clean, new format — no emoji)
    timeline_matches = re.findall(
        r"\[(\d+)%\]:\s*\n(.*?)(?=\[\d+%\]|\Z)",
        text, re.DOTALL
    )
    if timeline_matches:
        for pct_str, content in timeline_matches:
            block = content.strip()
            block = re.sub(r'\n\s*\n', '\n', block)
            lines_list = [l.strip() for l in block.splitlines()]
            prompt_text = "\n".join(lines_list)
            if prompt_text:
                prompts.append(prompt_text)
        if prompts:
            print(f"  -> Format: timeline [%] ({len(prompts)} sequences, clean)")
    else:
        # ── Format 3b: Timeline [X%] [📷 Image / ✨ Generated] (legacy)
        timeline_matches = re.findall(
            r"\[(\d+)%\]\s+\[(?:📷|✨).*?\]\s*(.*?)(?=\[\d+%\]|─|═|\Z)",
            text, re.DOTALL
        )
        if timeline_matches:
            for pct_str, content in timeline_matches:
                block = content.strip()
                block = re.sub(r'\n\s*\n', '\n', block)
                lines_list = [l.strip() for l in block.splitlines()]
                prompt_text = "\n".join(lines_list)
                if prompt_text:
                    prompts.append(prompt_text)
            if prompts:
                print(f"  -> Format: timeline [%] ({len(prompts)} sequences, legacy emoji)")

    return global_prompt, *_strip_meta(prompts)


def map_photos_to_segments(image_files: list[str], n_timeline: int, n_segments: int) -> dict:
    """
    Ordnet Bilder den JSON-Segmenten zu basierend auf ihrer Timeline-Position.

    Das Praefix (a_=0, b_=1, ...) kodiert den WebUI-Slot (0..9).
    Da nur die ersten n_timeline Slots aktiv sind, gilt: Praefix-Index = Timeline-Slot.
    Daraus wird die Timeline-%-Position berechnet und auf n_segments gemappt.

    Returns: {filename: json_segment_index}
    """
    if not image_files or n_timeline <= 0 or n_segments <= 0:
        return {}

    mapping = {}
    sorted_files = sorted(image_files)
    for seq_idx, fname in enumerate(sorted_files):
        # Extrahiere Slot-Index aus Praefix: a_cljc_.png → 0
        prefix = fname[0].lower() if fname else ''
        export_slot = ord(prefix) - ord('a')

        # Timeline-Slot bestimmen
        if 0 <= export_slot < n_timeline:
            # Praefix passt — dieses Foto gehoert an die export_slot-te Timeline-Position
            timeline_slot = export_slot
        else:
            # Fallback: sequentiell (Praefix ausserhalb des aktiven Bereichs)
            timeline_slot = seq_idx
            if timeline_slot >= n_timeline:
                timeline_slot = n_timeline - 1

        # Prozent-Position in der Timeline
        pct = round(timeline_slot * 100 / max(1, n_timeline - 1))

        # % → JSON-Segment-Index
        seg_idx = round(pct * (n_segments - 1) / 100)
        seg_idx = max(0, min(seg_idx, n_segments - 1))

        mapping[fname] = seg_idx
        print(f"  🖼️  Foto '{fname}' (Export-Slot {export_slot}, Timeline-Slot {timeline_slot}, ~{pct}%) → Segment {seg_idx + 1}")

    return mapping


def find_json_pairs() -> list[tuple[str, str]]:
    """
    Findet low.json und high.json im SCRIPT_DIR.
    Gibt Liste von (input_name, output_name) zurueck.
    """
    found = []
    for src_name, dst_name in JSON_PAIRS:
        src = SCRIPT_DIR / src_name
        if src.exists():
            found.append((str(src), dst_name))
        else:
            print(f"⚠️  {src_name} nicht gefunden in {SCRIPT_DIR}")
    if not found:
        raise FileNotFoundError(f"Keines der JSON-Paare gefunden: {[s for s,_ in JSON_PAIRS]}")
    return found


# ==========================================
# LORAS — Stärke + Aktivierungswörter aus loras.md
# ==========================================

def load_lora_map() -> dict:
    """
    Liest loras.md (Markdown-Tabelle: | Dateiname | Aktivierungswort | Stärke |).
    Returns: {dateiname_lower: {"activation": str, "strength": float}}
    """
    lora_map = {}
    path = SCRIPT_DIR / "loras.md"
    if not path.exists():
        print("⚠️  loras.md nicht gefunden — LoRA-Stärke/Aktivierung übersprungen.")
        return lora_map

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        fname, activation, strength = cells[0], cells[1], cells[2]
        if not fname or fname.lower() == "dateiname" or fname.startswith("---"):
            continue  # заголовок/разделитель таблицы
        try:
            strength = float(strength)
        except ValueError:
            strength = 1.0
        # activation может содержать примечания в скобках — берём только слова до "("
        act = activation.split("(")[0].strip()
        lora_map[fname.lower()] = {"activation": act, "strength": strength}
    print(f"📚 loras.md: {len(lora_map)} LoRAs geladen.")
    return lora_map


def apply_loras(workflow: dict, lora_map: dict, global_prompt: str) -> str:
    """
    Für alle LoraLoaderModelOnly-Nodes im Workflow:
      - widgets_values[1] (Stärke) aus loras.md setzen (falls bekannt).
      - Aktivierungswörter der verwendeten LoRAs sammeln.
    Gibt den aktualisierten global_prompt zurück (Aktivierungswörter angehängt).
    Der Prompt-Text selbst (Segmente) wird NICHT angefasst — nur global_prompt erweitert.
    """
    activation_words = []
    for node in workflow.get("nodes", []):
        if node.get("type") != "LoraLoaderModelOnly":
            continue
        widgets = node.get("widgets_values") or []
        if not widgets:
            continue
        lora_name = str(widgets[0])
        base_name = lora_name.split("\\")[-1].split("/")[-1]  # "ltx2\name.safetensors" → "name.safetensors"
        entry = lora_map.get(base_name.lower())
        if entry is None:
            print(f"  ⚠️ LoRA '{base_name}' nicht in loras.md — Stärke unverändert ({widgets[1] if len(widgets)>1 else 1}).")
            continue
        # ── Stärke setzen ──────────────────────────────────────────────
        if len(widgets) > 1:
            old = widgets[1]
            widgets[1] = entry["strength"]
            print(f"  🧠 LoRA '{base_name}': Stärke {old} → {entry['strength']}")
        # ── Aktivierungswort sammeln ───────────────────────────────────
        act = entry["activation"]
        if act and act.lower() not in ("kein trigger", "kein trigger-word", "n/a", "(kein"):
            activation_words.append(act)

    if activation_words:
        # Активационные слова добавляем в global_prompt отдельным блоком,
        # чтобы не затереть существующий текст промта.
        words = ", ".join(dict.fromkeys(activation_words))  # уникальные, сохраняя порядок
        trigger_block = f"\n\nLO-RA ACTIVATION: {words}"
        if global_prompt and trigger_block.strip() not in global_prompt:
            global_prompt = global_prompt.rstrip() + trigger_block
            print(f"  ✨ Aktivierungswörter in Global Prompt: {words}")
    return global_prompt


# ==========================================
# MAIN
# ==========================================

def main():
    print("🚀 Workflow-Updater — suche unbearbeiteten Projektordner...")

    # 1. Finde Arbeitsordner
    work_dir = find_work_folder()
    if work_dir is None:
        print("❌ Keine Projektordner in workflow/ gefunden.")
        print("   Exportiere zuerst einen Prompt (🖼️ Промт с фото / 💾 Export).")
        return

    print(f"📁 Verwende Ordner: '{work_dir.name}'")
    os.chdir(work_dir)

    # 2. Bilder + Prompts aus dem Ordner lesen
    image_files = find_images(work_dir)
    print(f"🖼️  Bilder ({len(image_files)}): {image_files}")

    prompt_path = work_dir / "video.txt"
    if not prompt_path.exists():
        prompt_path = work_dir / "novideo.txt"
    global_prompt, new_prompts, segment_weights = parse_prompt_file(prompt_path)
    print(f"📝 Global Prompt: {'✅' if global_prompt else '❌'}")
    print(f"📝 Sequenzen: {len(new_prompts)}")

    if not new_prompts:
        print("❌ Keine Sequenzen im Prompt. Abbruch.")
        return

    # 3. JSON-Paare finden
    try:
        json_pairs = find_json_pairs()
        print(f"✅ Gefunden: {[p[1] for p in json_pairs]}")
    except Exception as e:
        print(f"❌ {e}")
        return

    # 3b. LoRA-Daten laden (Stärke + Aktivierungswörter aus loras.md)
    lora_map = load_lora_map()

    # 4. Fuer jedes JSON: LTXDirector-Nodes aktualisieren
    for input_path, output_name in json_pairs:
        with open(input_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        print(f"\n{'='*50}\n⚙️  {Path(input_path).name} → {output_name}\n{'='*50}")

        # ── LoRA-Stärke setzen + Aktivierungswörter in Global Prompt ──
        # Выполняется ДО записи global_prompt в ноду, чтобы активационные
        # слова попали в итоговый промт. Сегментные промты не трогаются.
        global_prompt_eff = apply_loras(workflow, lora_map, global_prompt)

        nodes_updated = 0
        for node in workflow.get("nodes", []):
            # LTXDirector = LTX 2.0/2.3 (whatdreamscost); LTXDirectorCS25 = LTX 2.5 Director (CS)
            # Layout identisch: Widget 6 = timeline_data, 7 = local_prompts, 8 = segment_lengths
            if node.get("type") not in ("LTXDirector", "LTXDirectorCS25"):
                continue
    
            nodes_updated += 1
            print(f"\n⚙️  {node.get('type')} (ID: {node.get('id')})")
    
            widgets = node["widgets_values"]
    
            # ── timeline_data (JSON) aus Widget 6 laden ──────────────────
            timeline_data = json.loads(widgets[W_TIMELINE_DATA])
    
            if global_prompt_eff:
                timeline_data["global_prompt"] = global_prompt_eff
                print("  -> Global Prompt ✓")
    
            segments_list = timeline_data.get("segments", [])
            updated_prompts_for_ui = []
    
            n_segments_old = len(segments_list)
            n_timeline = len(new_prompts)
            print(f"  -> Workflow hat {n_segments_old} Segmente, Timeline hat {n_timeline} Slots")
    
            # ── Segment-Anzahl an Timeline anpassen ──────────────────────
            if n_timeline != n_segments_old:
                print(f"  🔧 Passe Segment-Anzahl an: {n_segments_old} → {n_timeline}")
                if n_timeline > n_segments_old:
                    # Expand: klone letztes Segment als Vorlage
                    template = segments_list[-1] if segments_list else {
                        "id": f"{int(time.time() * 1000)}",
                        "type": "text",
                    }
                    for i in range(n_segments_old, n_timeline):
                        new_seg = dict(template)
                        new_seg["id"] = f"{int(time.time() * 1000)}{i}"
                        new_seg.pop("imageFile", None)
                        new_seg.pop("imageB64", None)
                        segments_list.append(new_seg)
                else:
                    # Shrink: entferne ueberschuessige Segmente
                    segments_list = segments_list[:n_timeline]
    
            # Wichtig: nach Aenderung zurueckschreiben
            timeline_data["segments"] = segments_list
    
            # ── Start/Length gleichmaessig neu berechnen ────────────────
            # Gesamtlaenge bevorzugt aus normalDurationFrames (timeline_data),
            # sonst aus widgets[5] (duration_frames), sonst Fallback.
            total_duration = timeline_data.get("normalDurationFrames") or widgets[5] or 528
            if not total_duration and segments_list:
                last_start = segments_list[-1].get("start", 0)
                last_len   = segments_list[-1].get("length", 100)
                total_duration = last_start + last_len
    
            # ── Длины сегментов: DURATION_WEIGHT (1-5) от LLM → пропорционально, иначе поровну ──
            if segment_weights and len(segment_weights) == n_timeline and any(w for w in segment_weights):
                ws = [w if w and 1 <= w <= 5 else 1 for w in segment_weights]
                total_w = sum(ws)
                seg_lens = [max(1, total_duration * w // total_w) for w in ws]
                seg_lens[-1] = total_duration - sum(seg_lens[:-1])
            else:
                seg_len = max(1, total_duration // max(1, n_timeline))
                seg_lens = [seg_len] * n_timeline
                seg_lens[-1] = total_duration - seg_len * (n_timeline - 1)
            for i, seg in enumerate(segments_list):
                seg["start"]  = sum(seg_lens[:i])
                seg["length"] = seg_lens[i]
    
            # ── 1:1 Prompt-Zuordnung (Laengen sind jetzt gleich) ────────
            for i, segment in enumerate(segments_list):
                segment["prompt"] = new_prompts[i]
                updated_prompts_for_ui.append(new_prompts[i])
    
            # ── Foto → Segment (Timeline-Position) ──────────────────────
            photo_map = map_photos_to_segments(image_files, n_timeline, n_timeline)
    
            for i, segment in enumerate(segments_list):
                # Clear old image data first
                segment.pop("imageFile", None)
                segment.pop("imageB64", None)
    
                assigned_img = None
                for fname, seg_idx in photo_map.items():
                    if seg_idx == i:
                        assigned_img = fname
                        break
    
                if assigned_img:
                    segment["imageFile"] = assigned_img
                    segment["imageB64"] = f"/api/view?filename={assigned_img}&type=input&subfolder="
                    segment["type"] = "image"
                    print(f"  -> Seg {i+1}: '{assigned_img}' (Foto an Pos {i+1})")
                else:
                    segment["type"] = "text"
                    print(f"  -> Seg {i+1}: kein Bild")
    
            # ── Zurueckschreiben in die widgets_values ───────────────────
            timeline_data_str = json.dumps(timeline_data, ensure_ascii=False)
            local_prompts_str = " | ".join(updated_prompts_for_ui)
            segment_lengths_str = ",".join(str(seg["length"]) for seg in segments_list)
    
            widgets[W_TIMELINE_DATA]   = timeline_data_str
            widgets[W_LOCAL_PROMPTS]   = local_prompts_str
            widgets[W_SEGMENT_LENGTHS] = segment_lengths_str
    
            # ── WICHTIG: node["properties"] ist ein zweiter, separater
            # Datenspeicher, aus dem das LTXDirector-2.0-Widget beim Laden
            # in ComfyUI seinen Zustand rekonstruiert. Ohne diesen Sync
            # zeigt ComfyUI die alten (Template-)Werte an, obwohl
            # widgets_values bereits korrekt aktualisiert wurde.
            props = node.get("properties")
            if isinstance(props, dict):
                if global_prompt_eff:
                    props["global_prompt"] = global_prompt_eff
                props["local_prompts"]   = local_prompts_str
                props["segment_lengths"] = segment_lengths_str
                props["timeline_data"]   = timeline_data_str
                print("  -> properties[] synchronisiert ✓")
    
            print("  ✅ Fertig.")
    
        if nodes_updated == 0:
            print(f"  ⚠️  Kein LTXDirector-Node in {Path(input_path).name}")
        else:
            # Speichern IM Projektordner
            out_path = work_dir / output_name
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(workflow, f, indent=2, ensure_ascii=False)
            print(f"💾 Gespeichert: '{out_path}'")

    # 5. In used.md vermerken
    mark_used(work_dir.name)

    print(f"🎉 Fertig. Lade '{out_path.name}' in ComfyUI.")

if __name__ == "__main__":
    main()
