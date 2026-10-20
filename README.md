# 🎬 RAG — KI-Videoproduktion: Szenario → FLUX → LTX → ComfyUI

RAG-System mit Vektordatenbank (ChromaDB), lokalen LLMs (llama.cpp / KoboldCpp) und Cloud-APIs
(OpenAI, Groq, DeepSeek, NVIDIA, OpenRouter, Google, Mistral u. a.). Generiert
LTX-Video-Director-Prompts, FLUX-T2I-Prompts und ControlNet-Captions aus Text + Fotos —
inklusive Selbstkontrolle auf Fehler (2-Pass-Generator).

---

## 📁 Projektstruktur

```
RAG/
├── webui.py                # Gradio WebUI (Hauptinterface, http://127.0.0.1:7860)
├── rag_core.py             # Kern: Embeddings, ChromaDB, LLM-Backends, 2-Pass-Generator
├── rag_config.py           # Zentrale Konfiguration: Pfade, Ports, Timeouts, Cloud-Provider
├── rag_net.py              # HTTP-Helfer (Retries, Backoff, Endpunkt-Health)
├── rag_index.py            # Indizierung der Dokumente in ChromaDB
├── rag_query.py            # CLI-Chat (alternativ zur WebUI)
├── update_db.py            # Datenbank-Update
├── stats.py                # DB-Statistik
├── tokens.py               # Token-Verwaltung / Zähler
├── extract_to_md.py        # Konvertierung HTML → Markdown (Alt-Texte)
├── copilot.py              # Legacy: HTML-Post-Extraktor (zaebalo-Mirror)
├── index.py                # Legacy: Pagination-Fix für den HTML-Mirror
├── project_manager.py      # FLUX T2I Prompt-Generator (Projekt-Manager)
├── image_captioning.py     # Florence-2 / Qwen-VL Bildbeschreibung (Multi-Frame / Timeline)
├── zimage_generate.py      # Z-Image-Turbo Bild-Generator (via zimage.bat)
│
├── Workflow/               # ComfyUI-Integration & Vorlagen
│   ├── update_workflow.py    # JSON-Workflow-Updater (LTXDirector-Nodes)
│   ├── tripple.py            # Tripple-Workflow-Helfer
│   ├── grid_splitter.py      # Grid-Bild-Trenner
│   ├── video_00009.json      # Workflow-Vorlage (LTX Director)
│   ├── Tripple.json          # Tripple-Workflow-Vorlage
│   ├── LTX_2_5_Director_CStest.json         # LTX 2.5 Director Vorlage
│   ├── Text-to-Video Fast LTX2.5 Sync Audio & Video.json
│   ├── loras_triggerwords.md # LoRA-Katalog / Triggerwords
│   └── lora_catalog.json     # LoRA-Katalog (maschinenlesbar)
│
├── projects/               # FLUX-Projekt-Output (scenario_NAME_NN/)
├── tests/                  # pytest-Testsuite (rag_core, rag_config, rag_net, webui …)
├── docs/                   # Quelldokumente für die RAG-Datenbank (.txt/.md) — nicht committet
├── Songs/                  # Generierte Songtexte (Auto-Save) — nicht committet
├── model/                  # SentenceTransformer Modell (Embeddings) — nicht committet
├── florence-2-large/       # Florence-2 Vision-Modell — nicht committet
├── chroma_zaebalo/         # ChromaDB Vektordaten (Hauptbasis) — nicht committet
├── bookrag/                # Zweite ChromaDB (Bücher, optional via RAG_DB)
│
├── promt.md                # System-Prompt-Vorlage (LTX-Video-Prompt-Engineering)
├── settings.txt            # Sampling-Parameter & Server-Einstellungen (Tab-getrennt)
├── stylemusic.json         # 56 Musik-Stile (10 Kategorien) — nicht committet
├── cut_frames.txt          # ffmpeg/ffprobe-Snippets (Frames schneiden)
├── .env.example            # Vorlage für API-Keys → nach `.env` kopieren
├── requirements.txt        # Python-Abhängigkeiten
├── pyproject.toml          # Projektdefinition, pytest/ruff/mypy-Konfiguration
├── setup.bat               # Einmalig: Miniforge + Python 3.11 + venv
├── install_packages.bat    # Torch (CUDA 12.8) + requirements.txt
├── start.bat               # WebUI starten (portable Umgebung)
├── start-server.bat        # Nur llama-server starten
├── RAG.bat                 # CLI-Start mit Qwen2.5-Coder (llama-server Port 8080)
└── zimage.bat              # Z-Image-Turbo Generator starten
```

---

## ⚙️ Installation

### Systemvoraussetzungen

- Windows 10/11 (die `.bat`-Skripte sind für Windows; Python-Code läuft auch unter Linux/macOS)
- NVIDIA-GPU empfohlen (Torch wird mit CUDA 12.8 installiert; CPU funktioniert langsamer)
- ~15 GB Festplatte (Modelle: SentenceTransformer, Florence-2-large, GGUF-LLM)

### Portable (empfohlen)

```bat
setup.bat              :: Einmalig: Miniforge + Python 3.11-Umgebung in .\env
install_packages.bat   :: Torch (CUDA 12.8) + alle Pakete aus requirements.txt
start.bat              :: WebUI → http://127.0.0.1:7860
```

`start.bat` beendet automatisch eine alte WebUI-Instanz auf Port 7860 und startet
`webui.py` in der portablen Conda-Umgebung.

### Manuell (eigenes Python ≥ 3.10)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
copy .env.example .env      # API-Keys eintragen
python webui.py
```

CLI-Chat statt WebUI: `python rag_query.py`

---

## 🔌 LLM-Backends

Automatische Erkennung in dieser Reihenfolge:

| Priorität | Backend | Port/Status |
|---|---|---|
| 1 | **KoboldCpp** | Port 5001 |
| 2 | **llama-server** (llama.cpp) | Port 8080 (+ Extra-Ports, z. B. 8484, MTP 8510) |
| 3 | **Cloud-API** | Auto-Fallback auf ersten Key in `.env` |

### `.env` (API-Keys)

`.env` **nicht** committen — Vorlage ist `.env.example`:

```env
OPENAI_API_KEY=
GROQ_API_KEY=
DEEPSEEK_API_KEY=sk-...
NVIDIA_NIM_API_KEY=nvapi-...
OPENROUTER_API_KEY=sk-or-...
GOOGLE_API_KEY=
MISTRAL_API_KEY=
ODIROUTER_API_KEY=
AGNES_API_KEY=
```

Ein Key genügt — das System nimmt den ersten verfügbaren (Reihenfolge:
`rag_config.CLOUD_PROVIDERS`). In der WebUI kann das Backend/der Cloud-Provider
auch manuell gewählt werden. Optional konfigurierbar per `.env`: `RAG_DB`,
`RAG_COLLECTION`, `COMFY_INPUT_DIR`, `KOBOLD_PORT`, `LLAMA_PORT`, `MTP_PORT`,
`REMOTE_LLAMA`, `WEBUI_PORT`, Timeouts und Retry-Verhalten.

### `settings.txt` (Sampling & Server)

Tabulator-getrennt (`KEY<TAB>WERT`), wird beim Start geladen:

| Parameter | Default | Bedeutung |
|---|---|---|
| `MODEL_PATH` | *(leer)* | Pfad zur GGUF-Datei; gesetzt → llama-server startet automatisch |
| `temperature` | 0.85 | Sampling-Temperatur |
| `top_p` / `top_k` / `min_p` | 0.92 / 72 / 0.10 | Kern-/Top-k-Sampling |
| `repeat_penalty` | 1.25 | Wiederholungsstrafe |
| `presence_penalty` / `frequency_penalty` | 0.35 / 0.35 | Strafen für neue/wiederholte Tokens |
| `seed` | 0 | 0 = zufällig |
| `disable_thinking` | true | Reasoning-Gedanken des Modells unterdrücken |
| `ctx_size` | 8192–16384 | Kontextfenster (erste Zahl wird genutzt) |
| `n_gpu_layers` | 99 | 99 = alle Layer auf GPU |

Umgebungsvariable `RAG_DB` überschreibt den Pfad zur ChromaDB (Standard: `chroma_zaebalo/`).

---

## 🎛️ WebUI — Funktionen

### 📝 Text-Generierung

| Einstellung | Werte |
|---|---|
| Stil | Realismus / Düster / Humor / Bewusstseinsstrom |
| Modus | Literarisch / Analytisch |
| Sprache | RU, DE, EN, ES, PL, NL, FR |
| Länge | 50–700 Wörter |
| System-Prompt | Aus `.md`/`.json` laden (z. B. `promt.md`) oder direkt eingeben |

### 🎬 Szenario-Modus (LTX 2.3)

Wird «Формат LTX 2.3 → сценарий» gewählt:

| Neue Elemente | Funktion |
|---|---|
| **Рассказчик / Musik:Clip** | 70 Einträge: 14 Erzähler + 56 Musik-Stile aus `stylemusic.json` |
| **Director Mode** | Strukturierte Ausgabe: Director Node / Cinematographer / Audio / Dialogue |
| **🔢 Anzahl Szenen** | 1–5 (wird bei Fotos automatisch überschrieben) |
| **🎥 Kamera-Matrix** | 5×5 Grid: Dutch Angle, Dolly Zoom, Push-In, Low-Angle, POV × 5 Zeitabschnitte |

### 📸 Foto→Szene-Pipeline

```
🖼️ bis zu 4 Fotos
        ↓
🔍 Florence-2 beschreibt jedes Foto
        ↓
🎯 n_sequences = Anzahl Fotos (automatisch, ignoriert UI-Dropdown)
        ↓
✍️ Pass 1: LLM generiert pro Foto eine Szene
        ↓
🔎 Pass 2: Supervisor prüft:
   • Jedes Foto als eigene Szene?
   • Gleiche Motive → eigene Szenen?
   • Time-Progression logisch?
   • Atmosphere-Übergänge glatt?
   • Bei Fehlern → automatische Korrektur
```

### 🖼️ Timeline (Multi-Frame)

Im Akkordeon «Мульти-кадровый таймлайн»:

| Modus | Funktion |
|---|---|
| **Video** | 1–5 Bilder → Florence-2 + LLM → LTX 2.3 Timeline-Prompts |
| **Controlnet** | 1–5 Bilder → Pose + Atmosphere + Negative Prompt → `controlnet.txt` |

**Buttons Timeline:** ⚡ Generieren · 📋 Copy · 🌐 Übersetzen (RU) · 💾 Export (Bilder + Prompt → `Workflow/` & ComfyUI-Input) · 🔄 JSON → Workflow (führt `update_workflow.py` aus)

### 🎵 Song-Generator

Ausgewählter Musikstil + Thema → Songtext, automatischer Save als `Songs/<titel>_<datum>.txt`.

### 📁 Projekt-Manager (FLUX)

- Erstellt Projektordner `projects/scenario_NAME_NN/`
- Generiert **FLUX T2I-Prompts** im cinematischen Prosa-Stil (100–180 Wörter)
- Qualitäts-Boilerplate: `masterwork, masterpiece, best quality, ultra HD, 8k resolution, …`
- Speichert `scene_1.txt` … `scene_N.txt` + `project_meta.json`

### 💾 Export unter der Antwort

| Button | Ziel |
|---|---|
| 🖼️ **Промт с фото** | Bilder (`a_XXXX_.png`…) + Prompt als `video.txt` → `Workflow/` & ComfyUI-`input\` |
| 📝 **Промт без фото** | Nur Prompt als `novideo.txt` → beide Ordner |

Zusätzlich: JSON-Export für **Tripple**- und **LTX 2.5 First/Last-Frame**-Workflows.

---

## 🔄 ComfyUI-Workflow

```
1. 🖼️ Промт с фото  →  video.txt + Bilder in Workflow/
2. 🔄 JSON → Workflow  →  update_workflow.py:
   • Scannt Bilder alphabetisch
   • Parst video.txt (3 Formate: Delimiter / SEQUENCE / Timeline)
   • Updated LTXDirector-Nodes
   • Output: modified_LTX_Dir.json
3. JSON in ComfyUI laden und rendern
```

Vorlagen in `Workflow/`: `video_00009.json` (LTX Director), `Tripple.json`,
`LTX_2_5_Director_CStest.json`, `Text-to-Video Fast LTX2.5 Sync Audio & Video.json`.

---

## 🎵 Musik-Stile

`stylemusic.json` enthält **56 Stile** in **10 Kategorien**:

| Kategorie | Beispiele |
|---|---|
| Cinematic | soundtrack, emotional piano, dark ambient, epic orchestral, sci-fi synth… |
| Dark/Gothic | gothic rock, dark folk, slavic pagan, nordic folk, ritual ambient… |
| Rock/Metal | classic rock, heavy rock, industrial, grunge… |
| Electronic | synthwave, retrowave, techno, deep house, drum & bass… |
| Pop | soft pop, indie pop, electro-pop, acoustic pop… |
| Hip-Hop | trap, oldschool rap, rhythmic spoken… |
| World/Ethno | slavic folk, balkan, japanese shakuhachi, middle-eastern oud… |
| Emotional/Minimal | minimal piano, soft acoustic, whisper-style, intimate storytelling… |
| Futuristic | cyberpunk synth, glitch ambience, dystopian drone… |
| Uplifting | happy pop, upbeat acoustic, bright indie, cheerful folk… |

Wird automatisch in `narrator_map` geladen → LLM nutzt den jeweiligen Musik-Prompt.

---

## 🧠 Wie RAG funktioniert

```
Frage + Fotos (optional)
    ↓
SentenceTransformer → Embedding
    ↓
ChromaDB → Top-7 relevante Chunks
    ↓
Florence-2 → Bildbeschreibungen (wenn Fotos)
    ↓
Kontext + System-Prompt + Stil + Musik + Fotos
    ↓
LLM Pass 1 → Generierung
    ↓
LLM Pass 2 → Selbstkontrolle (nur Szenario-Modus)
    ↓
Strukturierte Ausgabe (---GLOBAL_PROMPT_START--- / ---SEQUENCE_N_START---)
```

Alle Pfade, Ports und Provider sind zentral in `rag_config.py` gebündelt;
HTTP-Aufrufe laufen über `rag_net.py` (Retries mit Backoff, Health-Checks).

---

## 🧪 Tests & Entwicklung

```bash
pip install -e ".[dev]"        # pytest, ruff, mypy
pytest                         # Testsuite in tests/
ruff check .                   # Linting (Line-Length 120)
mypy rag_config.py rag_net.py  # Typprüfung (Legacy-Module locker konfiguriert)
```

Konfigurationen liegen in `pyproject.toml` (`[tool.pytest]`, `[tool.ruff]`, `[tool.mypy]`).

---

## 📦 Abhängigkeiten

Pinned in `requirements.txt`:

```
sentence-transformers==5.5.1   # Embeddings (Modell in model/)
chromadb==1.5.9                # Vektordatenbank
transformers==4.57.6           # Florence-2 + Embeddings
gradio==5.50.0                 # WebUI
requests==2.34.2               # HTTP-APIs (KoboldCpp/Cloud)
beautifulsoup4==4.13.4         # HTML-Extraktion (extract_to_md.py)
python-dotenv==1.2.2           # .env-Laden
huggingface_hub==0.36.2        # Modell-Downloads
torch / torchvision / torchaudio  # separat via CUDA-Index (install_packages.bat)
```

---

## 🔧 Wartung

| Aufgabe | Wie |
|---|---|
| Dokumente hinzufügen | `.txt`/`.md` in `docs/` → Button «Обновить базу» (oder `python update_db.py`) |
| Neu indizieren | `python rag_index.py` |
| DB-Statistik | `python stats.py` |
| Embedding-Modell tauschen | Inhalt von `model/` ersetzen |
| DB zurücksetzen | `chroma_zaebalo/` löschen → neu indizieren |
| Cloud-API wechseln | `.env` editieren |
| Sampling anpassen | `settings.txt` editieren (Neustart nötig) |
| Musik-Stile erweitern | `stylemusic.json` editieren |
| Workflow-Vorlage | `Workflow/video_00009.json` ersetzen |

⚠️ **Sicherheit:** Die WebUI lauscht standardmäßig nur auf `127.0.0.1:7860`.
Bei Änderung auf `0.0.0.0` (LAN-Zugriff) hat die UI keine Authentifizierung —
Netzwerkzugriff nur hinter Firewall/VPN freigeben. `.env` niemals committen.

---

## ❓ Troubleshooting

| Problem | Lösung |
|---|---|
| Port 7860 belegt | `start.bat` killt alte Instanz automatisch; manuell: `netstat -ano \| findstr :7860` → `taskkill /f /pid <PID>` |
| „Kein Backend gefunden“ | KoboldCpp (Port 5001) oder llama-server (Port 8080) starten, sonst `.env`-Key prüfen |
| Florence-2 lädt nicht | Modellordner `florence-2-large/` vorhanden? Sonst via `huggingface_hub` herunterladen |
| Qwen-VL-Server startet nicht | Log `qwen_vl_server.log` prüfen (Port-Kollision via `MTP_PORT`/`LLAMA_EXTRA_PORTS` verschieben) |
| CUDA-Fehler | Torch aus `install_packages.bat` (cu128-Index) erneut installieren, Treiber aktualisieren |
| Veraltete Antworten aus DB | Dokumente in `docs/` aktualisiert? → «Обновить базу» |

---

## 🗂️ Hinweis zu Backup-/Legacy-Dateien

Im Repository liegen Entwicklungs-Backups (`*.bak*`, Log-/Dump-Dateien wie
`qwen_vl_server.log`, `webui_debug.log`, `1.md`). Sie sind nicht Teil des
Laufzeitsystems und können nach Sichtung gelöscht werden.
Große Laufzeit-artefakte (`model/`, `florence-2-large/`, `chroma_zaebalo/`,
`docs/`, `Songs/`, `stylemusic.json`, `.env`) sind bewusst **nicht** im Git-Repository —
siehe `.gitignore` bzw. obige Installationsanleitung.

## 📄 Lizenz

MIT-Lizenz — siehe [LICENSE](LICENSE).
