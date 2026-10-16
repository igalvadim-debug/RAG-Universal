#!/usr/bin/env python3
"""
grid_splitter.py - schneidet ein Bildgitter (Storyboard / Contact Sheet) sauber in Einzelbilder.

Ablauf:
  1. Qwen3-VL (llama-server mit mmproj) liefert die Rahmen (bbox) aller Panels samt Beschriftung.
  2. Die Rahmen werden per Pixel-Analyse auf die echten Kanten eingerastet (sauberer Schnitt).
  3. Panels werden von links nach rechts, von oben nach unten sortiert.
  4. Dateinamen: laufende Nummer + 3 zufaellige Buchstaben (1abc, 2grd, 3wek ...), PNG.
  5. Ausgabe nach D:\\claudeAgent\\RAG\\workflow\\gridN (erster freier Name laut used.md),
     der Ordnername wird in used.md eingetragen.

Aufruf:
  python grid_splitter.py "C:\\pfad\\bild.png"        (mehrere Bilder moeglich)
  python grid_splitter.py                             (fragt nach dem Pfad)
  Bild per Drag&Drop auf die Datei ziehen funktioniert ebenfalls.

Optionen: --ngl N (GPU-Layer, Standard: automatisch), --keep-server, --debug, --out-root PFAD
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import string
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
SERVER_EXE = Path(r"D:\Unsloth\C-claude\llama.cpp\llama-server.exe")   # wie in start.py
# Vision-Modelle: Modell + zugehoeriges mmproj starten immer als Paar
VL_SETS = {
    "gemma": (
        Path(r"D:\SillyTavernNeurogen\kobold\models\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"),
        Path(r"D:\SillyTavernNeurogen\kobold\models\mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
    ),
    "qwen": (
        Path(r"D:\ComfyUiVid\models\LLM\Qwen3-VL-4B-Instruct-abliterated.Q8_0.gguf"),
        Path(r"D:\ComfyUiVid\models\LLM\mmproj-F16.gguf"),
    ),
}
DEFAULT_VL = "qwen"   # Gemma-E2B findet die Rahmen nicht zuverlaessig -> mit --model gemma waehlbar
VL_PORT = 8485                      # eigener Port, stoert die 4 Dienste von start.py (8001-8003, 8484) nicht
REUSE_PORTS = (8485, 8080, 8484)          # laeuft dort schon ein Vision-Server, wird er benutzt
OUT_ROOT = Path(r"D:\claudeAgent\RAG\workflow")
SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_LOG = SCRIPT_DIR / "grid_splitter_server.log"

PROMPT = (
    "This image is a contact sheet / storyboard: there may be a title header, legends, colour swatches "
    "or a reference strip, then a grid of shot panels. Each panel consists of a picture plus the caption "
    "bar or text directly below or beside it (number, time range, title, camera, action). "
    "Detect every panel of the grid exactly once, INCLUDING its caption text. "
    "Do NOT include the overall title header, legends, colour swatches, reference strips or footers. "
    "Answer ONLY with a JSON array, ordered left to right and top to bottom: "
    '[{"n": 1, "bbox_2d": [x1, y1, x2, y2]}, ...] '
    "Coordinates are relative values from 0 to 1000 (x1,y1 = top-left corner, x2,y2 = bottom-right corner)."
)


PANELS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 24,
    "items": {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["n", "bbox_2d"],
        "additionalProperties": False,
    },
}


_log_cb = None   # optionaler Callback (z.B. fuer die Gradio-Oberflaeche)


def log(msg: str) -> None:
    print(msg, flush=True)
    if _log_cb is not None:
        try:
            _log_cb(msg)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------
def server_has_vision(port: int) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/props", timeout=2)
        if r.status_code != 200:
            return False
        return bool(r.json().get("modalities", {}).get("vision"))
    except Exception:
        return False


def free_vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def start_server(ngl_arg: int | None, vl: str) -> subprocess.Popen:
    VL_MODEL, VL_MMPROJ = VL_SETS[vl]
    for p, what in ((SERVER_EXE, "llama-server.exe"), (VL_MODEL, "Vision-Modell"), (VL_MMPROJ, "mmproj")):
        if not p.exists():
            raise RuntimeError(f"[FEHLER] {what} nicht gefunden: {p}")

    free = free_vram_mib()
    if ngl_arg is not None:
        ngl = ngl_arg
    else:
        # Modell (~4 GB) + mmproj + Kontext brauchen ~6 GB freien VRAM, sonst auf CPU ausweichen
        ngl = 99 if (free is None or free >= 6000) else 0
    log(f"[INFO] Freier VRAM: {free if free is not None else '?'} MiB -> GPU-Layer: {ngl}"
        + ("  (CPU-Modus, dauert laenger)" if ngl == 0 else ""))

    cmd = [
        str(SERVER_EXE),
        "-m", str(VL_MODEL),
        "--mmproj", str(VL_MMPROJ),
        "-c", "16384",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-np", "1",
        "-ngl", str(ngl),
        "-fa", "on",
        "--jinja",
        "--host", "127.0.0.1",
        "--port", str(VL_PORT),
        "--temp", "0.1",
        "--top-k", "20",
    ]
    cmd.append("--mmproj-offload" if ngl > 0 else "--no-mmproj-offload")

    log(f"[INFO] Starte llama-server ({VL_MODEL.name} + mmproj) auf Port {VL_PORT} ...")
    logf = open(SERVER_LOG, "w", encoding="utf-8", errors="replace")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)

    t0 = time.time()
    while time.time() - t0 < 360:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
            except Exception:
                pass
            raise RuntimeError(f"[FEHLER] llama-server ist beendet (Code {proc.returncode}). Log-Ende:\n{tail}")
        try:
            r = requests.get(f"http://127.0.0.1:{VL_PORT}/health", timeout=2)
            if r.status_code == 200:
                log(f"[OK] Server bereit nach {time.time() - t0:.0f}s")
                return proc
        except Exception:
            pass
        time.sleep(1.5)
    proc.terminate()
    raise RuntimeError("[FEHLER] llama-server wurde nicht rechtzeitig bereit.")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()


# ----------------------------------------------------------------------------
# Modellabfrage und Auswertung
# ----------------------------------------------------------------------------
def ask_model(port: int, img: Image.Image, attempt: int) -> tuple[str, tuple[int, int]]:
    W, H = img.size
    scale = min(1.0, 1600 / max(W, H))
    send = img.convert("RGB")
    if scale < 1:
        send = send.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    send.save(buf, "JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "temperature": 0.1 if attempt == 0 else 0.4,
        "max_tokens": 1200,
        "stream": False,
        # Grammatik-Zwang: verhindert Endlosschleifen und kaputtes JSON beim kleinen Modell
        "response_format": {"type": "json_schema", "json_schema": {"name": "panels", "strict": True, "schema": PANELS_SCHEMA}},
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=1200)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], send.size


def parse_boxes(text: str, sent_size: tuple[int, int], orig_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        data = json.loads(text[a:b + 1])
    except Exception:
        return []

    raw: list[list[float]] = []
    for it in data:
        box = None
        if isinstance(it, dict):
            box = it.get("bbox_2d") or it.get("bbox") or it.get("box")
        elif isinstance(it, (list, tuple)):
            box = it
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            raw.append([float(v) for v in box])
    if not raw:
        return []

    W, H = orig_size
    sw, sh = sent_size
    biggest = max(max(bx) for bx in raw)
    out = []
    for x1, y1, x2, y2 in raw:
        if biggest <= 1000.5:                      # relative Koordinaten 0..1000
            fx, fy = W / 1000.0, H / 1000.0
        else:                                      # Pixel des gesendeten Bildes
            fx, fy = W / sw, H / sh
        l, r = sorted((x1 * fx, x2 * fx))
        t, bt = sorted((y1 * fy, y2 * fy))
        out.append((int(round(l)), int(round(t)), int(round(r)), int(round(bt))))
    return out


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def validate(boxes, W, H) -> list[tuple[int, int, int, int]] | None:
    clean = []
    for l, t, r, b in boxes:
        l, t, r, b = max(0, l), max(0, t), min(W, r), min(H, b)
        if (r - l) < 0.05 * W or (b - t) < 0.03 * H:
            continue
        if any(iou((l, t, r, b), c) > 0.6 for c in clean):
            continue
        clean.append((l, t, r, b))
    if len(clean) < 2:
        return None
    # Gitterzellen sind annaehernd gleich gross - sonst ist die Antwort des Modells unbrauchbar
    ws = sorted(c[2] - c[0] for c in clean)
    hs = sorted(c[3] - c[1] for c in clean)
    mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
    if any(abs((c[2] - c[0]) - mw) > 0.25 * mw or abs((c[3] - c[1]) - mh) > 0.25 * mh for c in clean):
        return None
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            if iou(clean[i], clean[j]) > 0.2:
                return None
    return clean


# ----------------------------------------------------------------------------
# Kanten einrasten (sauberer Schnitt)
# ----------------------------------------------------------------------------
def _edge(profile: np.ndarray, pos: int, win: int) -> int:
    """Sucht im Fenster um pos die staerkste Helligkeitskante (die naechste unter den starken)."""
    n = len(profile)
    lo, hi = max(1, pos - win), min(n - 1, pos + win)
    if hi - lo < 2:
        return pos
    d = np.abs(np.diff(profile.astype(np.float32)))      # d[i] = Kante zwischen Pixel i und i+1
    seg = d[lo - 1:hi]                                   # Kanten an Position lo..hi
    peak = float(seg.max())
    if peak < 4.0:
        return pos
    idx = np.nonzero(seg >= 0.5 * peak)[0]
    k = idx[int(np.argmin(np.abs(idx - (pos - lo))))]
    return lo + int(k)


def bg_mask(img: Image.Image) -> np.ndarray:
    """Bool-Maske: True = Pixel hat Hintergrundfarbe (Farbe aus den Bildraendern)."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    k = 3
    strips = np.concatenate([
        arr[:k].reshape(-1, 3), arr[-k:].reshape(-1, 3),
        arr[:, :k].reshape(-1, 3), arr[:, -k:].reshape(-1, 3),
    ])
    bg = np.median(strips, axis=0)
    return np.abs(arr - bg).max(axis=2) < 16


def _gutter_edge(frac: np.ndarray, uni: np.ndarray, pos: int, win: int, low_side: bool) -> int | None:
    """Findet die Zwischenraum-Rinne (fast reiner Hintergrund) nahe pos und liefert die Panel-Kante daran."""
    n = len(frac)
    lo, hi = max(0, pos - win), min(n - 1, pos + win)
    g = frac[lo:hi + 1] >= 0.98
    runs, i = [], 0
    while i < len(g):
        if g[i]:
            j = i
            while j + 1 < len(g) and g[j + 1]:
                j += 1
            runs.append((lo + i, lo + j))
            i = j + 1
        else:
            i += 1
    runs = [r for r in runs if (r[1] - r[0] + 1) >= 3 or r[0] == 0 or r[1] == n - 1]
    if not runs:
        return None

    # Echte Rinne: beidseitig von einer gleichmaessigen Linie (Rahmen) oder dem Bildrand begrenzt.
    # Leerer Platz innerhalb einer Beschriftung hat auf der Textseite keine solche Linie.
    def flanked(r):
        a_ok = r[0] == 0 or bool(uni[r[0] - 1])
        b_ok = r[1] == n - 1 or bool(uni[r[1] + 1])
        return a_ok and b_ok

    good = [r for r in runs if flanked(r)]
    runs = good or runs

    def dist(r):
        return 0 if r[0] <= pos <= r[1] else min(abs(pos - r[0]), abs(pos - r[1]))

    r = min(runs, key=dist)
    if low_side:
        return None if r[1] >= hi else r[1] + 1
    return None if r[0] <= lo and lo > 0 else r[0]


def snap_box(gray: np.ndarray, mask: np.ndarray, box, W: int, H: int):
    """Rastet die vier Kanten ein: zuerst an der Hintergrund-Rinne, sonst an der staerksten Helligkeitskante."""
    l, t, r, b = box
    win_e = max(8, int(0.02 * max(W, H)))
    win_g = max(12, int(0.03 * max(W, H)))
    bh, bw = b - t, r - l
    y0, y1 = t + int(0.1 * bh), b - int(0.1 * bh)
    x0, x1 = l + int(0.1 * bw), r - int(0.1 * bw)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return box
    cols_g, rows_g = mask[y0:y1, :].mean(axis=0), mask[:, x0:x1].mean(axis=1)
    cols_e, rows_e = gray[y0:y1, :].mean(axis=0), gray[:, x0:x1].mean(axis=1)
    cols_u, rows_u = gray[y0:y1, :].std(axis=0) < 20, gray[:, x0:x1].std(axis=1) < 20

    def one(fr, un, pr, pos, low):
        e = _gutter_edge(fr, un, pos, win_g, low)
        return e if e is not None else _edge(pr, pos, win_e)

    nl, nr = one(cols_g, cols_u, cols_e, l, True), one(cols_g, cols_u, cols_e, r, False)
    nt, nb = one(rows_g, rows_u, rows_e, t, True), one(rows_g, rows_u, rows_e, b, False)
    if nr - nl < 0.5 * bw or nb - nt < 0.5 * bh:
        return box
    return (max(0, nl), max(0, nt), min(W, nr), min(H, nb))


def sort_boxes(boxes):
    heights = sorted(b[3] - b[1] for b in boxes)
    med_h = heights[len(heights) // 2]
    rows: list[list] = []
    for bx in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (bx[1] + bx[3]) / 2
        if rows and abs(cy - np.mean([(q[1] + q[3]) / 2 for q in rows[-1]])) < 0.5 * med_h:
            rows[-1].append(bx)
        else:
            rows.append([bx])
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


# ----------------------------------------------------------------------------
# Ordner / used.md / Dateinamen
# ----------------------------------------------------------------------------
def read_used(used_md: Path) -> set[str]:
    names: set[str] = set()
    if used_md.exists():
        for line in used_md.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(re.split(r"\s+", line)[0].lower())
    return names


def pick_folder(out_root: Path) -> Path:
    used = read_used(out_root / "used.md")
    n = 1
    while True:
        name = f"grid{n}"
        folder = out_root / name
        busy = folder.exists() and any(folder.iterdir())
        if name.lower() not in used and not busy:
            return folder
        n += 1


def append_used(used_md: Path, name: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    prefix = ""
    if used_md.exists() and used_md.stat().st_size > 0:
        with open(used_md, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) not in (b"\n", b"\r"):
                prefix = "\n"
    with open(used_md, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{prefix}{name}  # {stamp}\n")


def make_names(count: int) -> list[str]:
    names, seen = [], set()
    for i in range(1, count + 1):
        while True:
            suffix = "".join(random.choices(string.ascii_lowercase, k=3))
            if suffix not in seen:
                seen.add(suffix)
                break
        names.append(f"{i}{suffix}")
    return names


# ----------------------------------------------------------------------------
# Hauptablauf pro Bild
# ----------------------------------------------------------------------------
def process(path: Path, port: int, out_root: Path, debug: bool) -> tuple[Path, list[Path]]:
    log(f"\n=== {path} ===")
    img = Image.open(path)
    img.load()
    W, H = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mask = bg_mask(img)

    boxes = None
    for attempt in range(3):
        log(f"[INFO] Frage Qwen3-VL (Versuch {attempt + 1}/3) ...")
        text, sent = ask_model(port, img, attempt)
        if debug:
            log("[DEBUG] Antwort:\n" + text)
        boxes = validate(parse_boxes(text, sent, (W, H)), W, H)
        if boxes:
            break
        log("[WARN] Antwort unbrauchbar, neuer Versuch ...")
    if not boxes:
        raise RuntimeError("Das Modell lieferte keine brauchbaren Rahmen. Nichts gespeichert.")

    boxes = sort_boxes([snap_box(gray, mask, b, W, H) for b in boxes])
    log(f"[OK] {len(boxes)} Panels erkannt.")

    if debug:
        from PIL import ImageDraw
        dbg = img.convert("RGB")
        dr = ImageDraw.Draw(dbg)
        for i, b in enumerate(boxes, 1):
            dr.rectangle(b, outline=(0, 255, 0), width=3)
            dr.text((b[0] + 6, b[1] + 6), str(i), fill=(255, 255, 0))
        dbg.save(SCRIPT_DIR / "grid_splitter_debug.png")
        log(f"[DEBUG] Kontrollbild: {SCRIPT_DIR / 'grid_splitter_debug.png'}")

    folder = pick_folder(out_root)
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, box in zip(make_names(len(boxes)), boxes):
        fp = folder / f"{name}.png"
        img.convert("RGB").crop(box).save(fp, "PNG")
        saved.append(fp)
        log(f"   {name}.png  {box[2] - box[0]}x{box[3] - box[1]}")
    append_used(out_root / "used.md", folder.name)
    log(f"[FERTIG] Gespeichert in {folder}  (in used.md eingetragen)")
    return folder, saved


_run_lock = threading.Lock()


def split_image(image_path, out_root: Path = OUT_ROOT, vl: str = DEFAULT_VL,
                ngl: int | None = None, debug: bool = False, on_log=None) -> tuple[Path, list[Path]]:
    """API fuer andere Programme (z.B. webui.py): schneidet ein Gitter, gibt (Ordner, [Dateien]) zurueck.
    Startet bei Bedarf den Vision-Server selbst und beendet ihn danach wieder. Wirft RuntimeError bei Fehlern."""
    global _log_cb
    path = Path(image_path)
    if not path.is_file():
        raise RuntimeError(f"Datei nicht gefunden: {path}")
    with _run_lock:
        _log_cb = on_log
        proc = None
        try:
            out_root = Path(out_root)
            out_root.mkdir(parents=True, exist_ok=True)
            port = next((p for p in REUSE_PORTS if server_has_vision(p)), None)
            if port is not None:
                log(f"[INFO] Vision-Server auf Port {port} gefunden, wird benutzt.")
            else:
                proc = start_server(ngl, vl)
                port = VL_PORT
            return process(path, port, out_root, debug)
        finally:
            stop_server(proc)
            _log_cb = None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Bildgitter per Qwen3-VL in Einzelbilder schneiden")
    ap.add_argument("images", nargs="*", help="Pfad(e) zu den Gitter-Bildern")
    ap.add_argument("--ngl", type=int, default=None, help="GPU-Layer (Standard: automatisch)")
    ap.add_argument("--model", choices=sorted(VL_SETS), default=DEFAULT_VL, help="Vision-Modell (Standard: qwen)")
    ap.add_argument("--keep-server", action="store_true", help="Server nach dem Lauf nicht beenden")
    ap.add_argument("--debug", action="store_true", help="Modellantwort und Kontrollbild ausgeben")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="Basisordner (Standard: workflow)")
    args = ap.parse_args()

    images = [Path(p.strip('"')) for p in args.images]
    if not images:
        p = input("Pfad zum Gitter-Bild: ").strip().strip('"')
        if not p:
            return
        images = [Path(p)]
    for p in images:
        if not p.is_file():
            raise SystemExit(f"[FEHLER] Datei nicht gefunden: {p}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    port = next((p for p in REUSE_PORTS if server_has_vision(p)), None)
    proc = None
    if port is not None:
        log(f"[INFO] Vision-Server auf Port {port} gefunden, wird benutzt.")
    else:
        try:
            proc = start_server(args.ngl, args.model)
        except RuntimeError as e:
            raise SystemExit(f"[FEHLER] {e}")
        port = VL_PORT

    ok = True
    try:
        for p in images:
            try:
                process(p, port, out_root, args.debug)
            except RuntimeError as e:
                log(f"[FEHLER] {e}")
                ok = False
    finally:
        if proc is not None and not args.keep_server:
            stop_server(proc)

    if "PROMPT" not in os.environ:       # per Doppelklick / Drag&Drop gestartet -> Fenster offen halten
        input("\nEnter zum Beenden ...")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
