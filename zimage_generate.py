"""
zimage_generate.py — separater Bild-Generator mit Z-Image-Turbo (GGUF).

Läuft mit dem ComfyUi-Python (venv mit diffusers + torch):
    C:\\ComfyUi\\venv\\Scripts\\python.exe zimage_generate.py "DEIN PROMPT" [Optionen]

Vor der Bildgenerierung wird llama-server.exe gestoppt (VRAM freigeben),
außer --keep-llm ist gesetzt.

Optionen:
    --height 1024   Bildhöhe
    --width  1024   Bildbreite
    --seed   42     Zufalls-Seed
    --steps  9      Inferenz-Schritte (Turbo: 9)
    --out    zimage.png   Ausgabedatei
    --keep-llm       llama-server NICHT stoppen
"""

import argparse
import subprocess
import sys
from pathlib import Path

GGUF_CANDIDATES = [
    Path(r"D:\ComfyUiCuda\models\checkpoints\z-image-turbo-Q8_0.gguf"),
    Path(r"D:\ComfyUiCuda\models\checkpoints\z_image_turbo-Q8_0.gguf"),
    Path(r"C:\ComfyUi\models\checkpoints\z-image-turbo-Q8_0.gguf"),
    Path(r"C:\ComfyUi\models\checkpoints\z_image_turbo-Q8_0.gguf"),
]


def find_gguf() -> Path | None:
    for p in GGUF_CANDIDATES:
        if p.exists():
            return p
    return None


def free_vram():
    """Stoppt llama-server, damit das Bildmodell in den VRAM passt."""
    try:
        r = subprocess.run(
            ["taskkill", "/f", "/im", "llama-server.exe"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print("[ZImage] llama-server gestoppt — VRAM freigegeben.")
        else:
            print("[ZImage] llama-server lief nicht (oder konnte nicht gestoppt werden).")
    except Exception as e:
        print(f"[ZImage] Warnung: llama-server konnte nicht gestoppt werden: {e}")


def main():
    ap = argparse.ArgumentParser(description="Z-Image-Turbo Bildgenerator (separates Skript)")
    ap.add_argument("prompt", nargs="?", default="",
                    help='Bild-Prompt in Anführungszeichen, z.B. "Eine Frau im roten Hanfu"')
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--out", default="zimage.png")
    ap.add_argument("--keep-llm", action="store_true",
                    help="llama-server NICHT stoppen (VRAM muss dann reichen)")
    args = ap.parse_args()

    # Prompt: Argument > zimage_prompt.txt > interaktive Abfrage
    prompt = (args.prompt or "").strip()
    if not prompt:
        pfile = Path(__file__).parent / "zimage_prompt.txt"
        if pfile.exists():
            prompt = pfile.read_text(encoding="utf-8").strip()
    if not prompt:
        print("[ZImage] Kein Prompt als Argument übergeben.")
        print("  Beispiele:")
        print('    python zimage_generate.py "Eine Frau im roten Hanfu"')
        print('    zimage.bat "Ein Drache über der Stadt bei Nacht"')
        print()
        try:
            prompt = input("Prompt eingeben (Enter = Beispiel): ").strip()
        except EOFError:
            prompt = ""
        if not prompt:
            prompt = ("Young Chinese woman in red Hanfu, intricate embroidery. "
                      "Impeccable makeup, red floral forehead pattern. "
                      "Elaborate high bun, golden phoenix headdress. "
                      "Soft-lit outdoor night background, silhouetted tiered pagoda, "
                      "blurred colorful distant lights.")
            print("\n[ZImage] Nutze Beispiel-Prompt.")
    print(f"[ZImage] Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

    gguf = find_gguf()
    if gguf is None:
        print("[ZImage] FEHLER: z-image-turbo-Q8_0.gguf nicht gefunden "
              "(gesucht in D:\\ComfyUiCuda und C:\\ComfyUi checkpoints).")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("[ZImage] FEHLER: CUDA ist nicht verfügbar — kann kein Bild generieren.")
        sys.exit(1)

    # LLM aus dem Speicher entladen, bevor das Bildmodell geladen wird
    if not args.keep_llm:
        free_vram()
        torch.cuda.empty_cache()

    from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

    print(f"[ZImage] Lade GGUF: {gguf}")
    transformer = ZImageTransformer2DModel.from_single_file(
        str(gguf),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )

    print("[ZImage] Lade Pipeline (Tongyi-MAI/Z-Image-Turbo)...")
    pipeline = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        dtype=torch.bfloat16,
    ).to("cuda")

    # Attention-Backend (optional; bester Effekt beim Turbo-Modell)
    for backend in ("_sage_qk_int8_pv_fp16_triton", "flash", "_flash_3"):
        try:
            pipeline.transformer.set_attention_backend(backend)
            print(f"[ZImage] Attention-Backend aktiviert: {backend}")
            break
        except Exception:
            continue

    print(f"[ZImage] Generiere {args.width}x{args.height}, seed={args.seed}, steps={args.steps} ...")
    image = pipeline(
        prompt=prompt,
        num_inference_steps=args.steps,   # Turbo: 9 Steps → 8 DiT-Forwards
        guidance_scale=0.0,               # Turbo: guidance 0
        height=args.height,
        width=args.width,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]

    out_path = Path(args.out)
    image.save(out_path)
    print(f"[ZImage] ✅ Bild gespeichert: {out_path.resolve()}")


if __name__ == "__main__":
    main()
