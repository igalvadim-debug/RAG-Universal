# 🎬 RAG — KI-Videoproduktion: Szenario → FLUX → LTX → ComfyUI

RAG-System mit Vektordatenbank (ChromaDB), lokalen LLMs (llama.cpp / KoboldCpp) und Cloud-APIs (OpenAI, Google, DeepSeek, NVIDIA, OpenRouter, Groq). Generiert LTX-Video-Director-Prompts, FLUX-T2I-Prompts und ControlNet-Captions aus Text + Fotos — inklusive Selbstkontrolle auf Fehler.

---

## 📁 Struktur

```
RAG/
├── webui.py              # Gradio WebUI (Hauptinterface)
├── rag_core.py           # Kern: Embeddings, Chroma, LLM, 2-Pass-Generator
├── rag_index.py          # Indizierung in ChromaDB
├── rag_query.py          # CLI-Chat
├── update_db.py          # Datenbank-Update
├── stats.py              # DB-Statistik
├── tokens.py             # Token-Verwaltung
├── extract_to_md.py      # Konvertierung → Markdown
├── project_manager.py    # FLUX T2I Prompt-Generator
├── image_captioning.py   # Florence-2 Bildbeschreibung
│
├── workflow/             # ComfyUI-Integration
│   ├── update_workflow.py  # JSON-Workflow-Updater
│   ├── video_00009.json    # Workflow-Vorlage
│   └── video.txt           # Generierte Prompts
│
├── stylemusic.json       # 56 Musik-Stile (10 Kategorien)
├── florence-2-large/     # Florence-2 Modell
├── model/                # SentenceTransformer (Embeddings)
├── chroma_zaebalo/       # ChromaDB Vektordaten
├── docs/                 # Quelldokumente
├── projects/             # FLUX-Projekt-Output
│
├── .env                  # API-Keys
├── requirements.txt
├── start.bat / setup.bat # Portable Start
└── RAG.bat               # CLI-Start
```

---

## ⚙️ Installation

### Portable (empfohlen)

```bat
setup.bat    # Einmalig: Miniforge + Python 3.11 + Pakete
start.bat    # WebUI → http://127.0.0.1:7860
```

### Manuell

```bash
pip install -r requirements.txt
python webui.py
```

---

## 🔌 LLM-Backends

Automatische Erkennung in dieser Reihenfolge:

| Priorität | Backend | Port/Status |
|---|---|---|
| 1 | **KoboldCpp** | Port 5001 |
| 2 | **llama-server** | Port 8080 |
| 3 | **Cloud-API** | Auto-Fallback auf ersten Key in `.env` |

### `.env`

```env
DEEPSEEK_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
NVIDIA_NIM_API_KEY=nvapi-...
OPENAI_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
```

Ein Key genügt — das System nimmt den ersten verfügbaren.

---

## 🎛️ WebUI — Funktionen

### 📝 Text-Generierung

| Einstellung | Werte |
|---|---|
| Stil | Realismus / Düster / Humor / Bewusstseinsstrom |
| Modus | Literarisch / Analytisch |
| Sprache | RU, DE, EN, ES, PL, NL, FR |
| Länge | 50–700 Wörter |
| System-Prompt | Aus `.md`/`.json` laden oder direkt eingeben |

### 🎬 Szenario-Modus (LTX 2.3)

Wird «Формат LTX 2.3 → сценарий» gewählt:

| Neue Elemente | Funktion |
|---|---|
| **Рассказчик / Musik:Clip** | 70 Einträge: 14 Erzähler + 56 Musik-Stile aus `stylemusic.json` |
| **Director Mode** | Strukturierte Ausgabe: Director Node / Cinematographer / Audio / Dialogue |
| **🔢 Anzahl Szenen** | 1–5 (wird bei Fotos automatisch überschrieben) |
| **🎥 Kamera-Matrix** | 5×5 Grid: Dutch Angle, Dolly Zoom, Push-In, Low-Angle, POV × 5 Zeitabschnitte |

### 📸 Foto→Szene-Pipeline (NEU)

```
🖼️ 4 Fotos (2×Paris, 1×Miami, 1×Ballett)
        ↓
🔍 Florence-2 beschreibt jedes Foto
        ↓
🎯 n_sequences = 4 (automatisch, ignoriert UI-Dropdown)
   SEQUENCE 1 = Foto 1 (Paris)
   SEQUENCE 2 = Foto 2 (Paris)
   SEQUENCE 3 = Foto 3 (Miami)
   SEQUENCE 4 = Foto 4 (Ballett)
        ↓
✍️ Pass 1: LLM generiert 4 Szenen
        ↓
🔎 Pass 2: Supervisor prüft:
   • Jedes Foto als eigene Szene?
   • 2×Paris → 2 Szenen Paris?
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

**Buttons Timeline:**
- ⚡ Generieren
- 📋 Copy
- 🌐 Übersetzen (RU)
- 💾 Export (Bilder + Prompt → `workflow/` & `ComfyUiVid\input\`)
- 🔄 JSON → Workflow (führt `update_workflow.py` aus)

### 📁 Projekt-Manager (FLUX)

- Erstellt Projektordner `projects/scenario_NAME_NN/`
- Generiert **FLUX T2I-Prompts** im cinematischen Prosa-Stil (100–180 Wörter)
- Qualitäts-Boilerplate: `masterwork, masterpiece, best quality, ultra HD, 8k resolution, …`
- Speichert `scene_1.txt` … `scene_N.txt` + `project_meta.json`

### 💾 Export unter der Antwort

| Button | Ziel |
|---|---|
| 🖼️ **Промт с фото** | Bilder (`a_XXXX_.png`…) + Prompt als `video.txt` → `workflow/` & `ComfyUiVid\input\` |
| 📝 **Промт без фото** | Nur Prompt als `novideo.txt` → beide Ordner |

---

## 🔄 ComfyUI-Workflow

```
1. 🖼️ Промт с фото → video.txt + Bilder in workflow/
2. 🔄 JSON → Workflow  → update_workflow.py:
   • Scannt Bilder alphabetisch
   • Parst video.txt (3 Formate: Delimiter / SEQUENCE / Timeline)
   • Updated LTXDirector-Nodes
   • Output: modified_LTX_Dir.json
3. In ComfyUI laden
```

---

## 🎵 Musik-Stile

`stylemusic.json` enthält **56 Stile** in **10 Kategorien**:

| Kategorie | Stile |
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

---

## 📦 Abhängigkeiten

```
sentence-transformers   # Embeddings (model/)
chromadb                # Vektordatenbank
gradio                  # WebUI
requests                # HTTP-APIs
Pillow                  # Bildverarbeitung
torch                   # Florence-2
transformers            # Florence-2 + Embeddings
```

---

## 🔧 Wartung

| Aufgabe | Wie |
|---|---|
| Dokumente hinzufügen | `.txt`/`.md` in `docs/` → «Обновить базу» |
| Embedding-Modell tauschen | `model/` ersetzen |
| DB zurücksetzen | `chroma_zaebalo/` löschen → neu indizieren |
| Cloud-API wechseln | `.env` editieren |
| Musik-Stile erweitern | `stylemusic.json` editieren |
| Workflow-Vorlage | `workflow/video_00009.json` ersetzen |
