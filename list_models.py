"""Modelle von der Platte auflisten -> ../docs/models.txt.

Laeuft beim Start der Oberflaeche (mm_ui.refresh_registry) und von Hand:

    venv\\Scripts\\python.exe docs\\list_models.py

Geschrieben wird immer die docs/models.txt NEBEN diesem Skript, unabhaengig vom
aktuellen Arbeitsverzeichnis. Bereits eingetragene Trigger-Anhaenge
('  ->  Trigger: ...') bleiben erhalten - sonst wuerde jeder Lauf die
gepflegten Triggerwoerter aus models.md wegwerfen.
"""
import os
from pathlib import Path

# Zu durchsuchende Modellordner (Wurzeln, nicht die Unterordner selbst).
FOLDERS_TO_SCAN = [
    r"D:\ComfyUi\models",
    r"D:\ComfyUiCuda\models",
    r"C:\ai_diffusion\models",
    r"D:\ComfyUiVideo\models",
    r"C:\Users\Startklar\.lmstudio\models\lmstudio-community",
    r"D:\SillyTavernAiO\KoboldCPP\models",
    r"D:\claudeAgent\Coder\models",
]

# Nur diese Endungen sind Modelle (case-insensitive).
EXTENSIONS = {".gguf", ".safetensors", ".pt", ".pth", ".ckpt", ".sft", ".bin"}

# Node-/Umgebungsmuell, der nie ein Modell ist.
SKIP_DIRS = {"custom_nodes", "venv", "__pycache__", ".cache", "node_modules",
             ".git", ".disabled", "text_embed_cache"}

OUTPUT_FILE = Path(__file__).resolve().with_name("models.txt")
TRIGGER_MARK = "  \u2192  "


def _trigger_tails(path: Path) -> dict:
    """Bereits eingetragene Trigger-Anhaenge aus der alten models.txt."""
    tails = {}
    if not path.exists():
        return tails
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        full, sep, tail = raw.partition("\u2192")
        if sep and full.strip():
            tails[full.strip().replace("/", "\\").lower()] = tail.strip()
    return tails


def find_models():
    """Alle Modelldateien sammeln, Trigger uebernehmen und models.txt schreiben."""
    old = _trigger_tails(OUTPUT_FILE)
    found: list[str] = []
    seen: set[str] = set()
    for folder_str in FOLDERS_TO_SCAN:
        folder = Path(folder_str)
        if not folder.is_dir():
            print(f"[!] Ordner fehlt, uebersprungen: {folder_str}")
            continue
        print(f"[i] Scanne: {folder_str}")
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
            for name in files:
                p = Path(root) / name
                if p.suffix.lower() not in EXTENSIONS:
                    continue
                full = str(p)
                key = full.replace("/", "\\").lower()
                if key in seen:
                    continue
                seen.add(key)
                tail = old.get(key)
                found.append(f"{full}{TRIGGER_MARK}{tail}" if tail else full)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_FILE.with_name(OUTPUT_FILE.name + ".tmp")
    tmp.write_text("\n".join(sorted(found, key=str.lower)) + ("\n" if found else ""),
                   encoding="utf-8")
    os.replace(tmp, OUTPUT_FILE)
    print(f"[OK] {len(found)} Modell(e) -> {OUTPUT_FILE}")
    return len(found)


if __name__ == "__main__":
    find_models()
