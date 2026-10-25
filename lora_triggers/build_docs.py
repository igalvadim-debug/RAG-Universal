import os, json, re, shutil, collections

BASE = r"D:\claudeAgent\RAG\workflow\lora_triggers"
RAW = os.path.join(BASE, "loras_raw.json")
LORAS_MD = r"D:\claudeAgent\RAG\workflow\loras.md"
RAG_MD = r"D:\claudeAgent\RAG\loras_triggerwords.md"
MODELS_MD = r"D:\claudeAgent\Hermes\models.md"
ROOTS = {"D": r"D:\ComfyUiCuda\models\loras", "C": r"C:\ai_diffusion\models\loras"}

OVR = {
    "ltx2-squish_trigger word_squish it.safetensors": ("squish it", "Dateiname + HF Comfy-Org/ltx-2 (Squish-LoRA)"),
    "nsfw_lora_wan_1.3b.safetensors": ("(kein Trigger)", "NSFW-API/NSFW_Wan_1.3b, kein Trigger angegeben"),
    "Klein-consistency.safetensors": ("(kein Trigger nötig)", "Flux2-Klein-9B-Consistency, laut Autor ohne Cue-Wörter"),
    "LTX-2.3-Licon-MSR-V1.safetensors": ("(kein Trigger, braucht ComfyUI-Licon-MSR Node)", "LiconStudio MSR"),
    "LTX-2.3-Licon-MSR-V2.safetensors": ("(kein Trigger, braucht ComfyUI-Licon-MSR Node)", "LiconStudio MSR V2"),
}

def clean(s):
    return re.sub(r"\s+", " ", str(s).replace("|", "/")).strip()

def short(s, n):
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"

def parse_old(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path, encoding="utf-8-sig"):
        if line.startswith("|") and not line.startswith("|---") and "Dateiname" not in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3:
                d[cells[0]] = (cells[1], cells[2])
    return d

def backup(p):
    b = p + ".bak_20261010"
    if os.path.exists(p) and not os.path.exists(b):
        shutil.copy2(p, b)

def main():
    raw = json.load(open(RAW, encoding="utf-8"))
    hfp = os.path.join(BASE, "hf_raw.json")
    hf = json.load(open(hfp, encoding="utf-8")) if os.path.exists(hfp) else {}
    backup(LORAS_MD)
    backup(MODELS_MD)
    old = parse_old(LORAS_MD if not os.path.exists(LORAS_MD + ".bak_20261010") else LORAS_MD + ".bak_20261010")

    rows = []
    stats = collections.Counter()
    for e in raw:
        name = e["file"]
        path = e["path"]
        drive = path[0].upper()
        rel = os.path.relpath(os.path.dirname(path), ROOTS[drive])
        folder = "%s: %s" % (drive, "(root)" if rel == "." else rel)
        o = old.get(name)
        cv = e.get("civitai") or {}
        trig, src = "", ""
        if name in OVR:
            trig, src = OVR[name]
        elif o and o[0]:
            trig, src = clean(o[0]), "loras.md (Hash-Abgleich)"
        elif cv.get("trainedWords"):
            words = []
            for w in cv["trainedWords"]:
                w = clean(w)
                if w and w not in words:
                    words.append(w)
            trig, src = " / ".join(words), "CivitAI"
        hfe = hf.get(name)
        if not trig and hfe and hfe.get("triggers"):
            trig = " / ".join(clean(t) for t in hfe["triggers"])
            src = "HuggingFace"
        info = ""
        if cv.get("modelId"):
            info = "%s / %s [%s]" % (cv.get("modelName"), cv.get("versionName"), cv.get("baseModel"))
        if hfe:
            info = (info + " " if info else "") + "HF: " + hfe["repo"]
        tags = [t for t in (e.get("meta", {}).get("top_tags") or []) if len(t) <= 40][:3]
        if not trig and tags:
            info = (info + " " if info else "") + "Training-Tags: " + ", ".join(tags)
        if not trig and not src:
            if cv.get("modelId"):
                src = "CivitAI (kein Trigger)"
            elif hfe:
                src = "HuggingFace (kein Trigger im README)"
            else:
                src = "nicht gefunden"
        strength = o[1] if o else "–"
        stats["gesamt"] += 1
        if trig and not trig.startswith("("):
            stats["mit Trigger"] += 1
        rows.append({"file": name, "path": path, "folder": folder, "trig": trig, "strength": strength, "src": src, "info": clean(info)})

    rows.sort(key=lambda r: (0 if r["folder"].endswith("ltx2") and r["folder"].startswith("D") else 1, r["folder"].lower(), r["file"].lower()))
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(r["folder"], []).append(r)

    out = []
    out.append("# LoRA Übersicht mit Aktivierungswörtern (Trigger-Words)\n")
    out.append("Stand: 2026-10-10. Quellen: SHA256-Hash-Abgleich mit CivitAI (API), früherer Abgleich in dieser Datei (loras.md), Websuche (HuggingFace/CivitAI/ComfyUI), Dateiname.")
    out.append("Ordner: D: = D:\\ComfyUiCuda\\models\\loras, C: = C:\\ai_diffusion\\models\\loras.")
    out.append("Aktivierungswort „–“ = kein Trigger nötig oder nicht auffindbar (Style-/Enhancer-/Utility-/Speed-LoRAs). Stärke „–“ = nicht bekannt.")
    out.append("Rohdaten (alle Trigger, Hashes, Links): D:\\claudeAgent\\RAG\\workflow\\lora_triggers\\loras_raw.json\n")
    out.append("Gesamt: %d LoRAs, davon %d mit Aktivierungswort.\n" % (stats["gesamt"], stats["mit Trigger"]))
    for folder, items in groups.items():
        out.append("\n## %s\n" % folder)
        out.append("| Dateiname | Aktivierungswort | Stärke | Quelle / Info |")
        out.append("|---|---|---|---|")
        for r in items:
            t = short(r["trig"], 160) if r["trig"] else "–"
            extra = r["src"] + ((" – " + r["info"]) if r["info"] else "")
            out.append("| %s | %s | %s | %s |" % (r["file"], t, r["strength"], extra))
    out.append("\n## Aus der alten Liste (liegen in latent_upscale_models, nicht im LoRA-Ordner)\n")
    out.append("| Dateiname | Aktivierungswort | Stärke | Quelle / Info |")
    out.append("|---|---|---|---|")
    out.append("| ltx-2.3-spatial-upscaler-x2-1.1.safetensors | – | 1.0 | loras.md (alt), Latent-Upscaler, kein Trigger |")
    out.append("| ltx-2.3-temporal-upscaler-x2-1.0.safetensors | – | 1.0 | loras.md (alt), Latent-Upscaler, kein Trigger |")
    text = "\n".join(out) + "\n"
    open(LORAS_MD, "w", encoding="utf-8", newline="\n").write(text)
    os.makedirs(os.path.dirname(RAG_MD), exist_ok=True)
    open(RAG_MD, "w", encoding="utf-8", newline="\n").write(text)

    # models.md annotieren
    mp = {}
    for r in rows:
        mp[r["path"].lower()] = short(r["trig"], 110) if r["trig"] else "–"
    raw_bytes = open(MODELS_MD, "rb").read()
    bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    content = raw_bytes.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in content else "\n"
    lines = content.split(eol)
    changed = 0
    new_lines = []
    for l in lines:
        base = re.sub(r"\s+→\s+Trigger:.*$", "", l)
        if base.lower().strip() in mp:
            base = base + "  →  Trigger: " + mp[base.lower().strip()]
            changed += 1
        new_lines.append(base)
    data = eol.join(new_lines)
    open(MODELS_MD, "wb").write((b"\xef\xbb\xbf" if bom else b"") + data.encode("utf-8"))
    print("loras.md/RAG: %d Zeilen, models.md annotiert: %d, mit Trigger: %d" % (stats["gesamt"], changed, stats["mit Trigger"]))
    missing_old = [k for k in old if k not in {r["file"] for r in rows}]
    print("alte Einträge ohne Datei im Ordner:", missing_old)

if __name__ == "__main__":
    main()
