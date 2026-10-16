"""
rag_config.py — единая конфигурация RAG-проекта (пути, порты, таймауты, API-ключи).

Все «магические числа» и пути собраны здесь в dataclass-ах. Модули rag_core.py,
webui.py и Workflow/* берут значения отсюда. Переопределение — через переменные
окружения (см. .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    """Return the stripped environment value, or default if unset or blank."""
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _scheme(host: str) -> str:
    """http:// для loopback-хостов (локальные llama-server/KoboldCpp), https:// для остальных."""
    h = host.strip().lower()
    if h.startswith("["):  # IPv6-литерал вида [::1]:port
        h = h[1:h.find("]")] if "]" in h else h
    else:
        h = h.split(":")[0]  # host или host:port
    if h in ("127.0.0.1", "localhost", "::1") or h.startswith("127."):
        return "http"
    return "https"


# ──────────────────────────────────────────────
# ПУТИ
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Paths:
    model_dir: Path = BASE_DIR / "model"
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env("RAG_DB", str(BASE_DIR / "chroma_zaebalo")))
    )
    bookrag_dir: Path = BASE_DIR / "bookrag"
    docs_dir: Path = BASE_DIR / "docs"
    songs_dir: Path = BASE_DIR / "Songs"
    projects_dir: Path = BASE_DIR / "projects"
    env_file: Path = BASE_DIR / ".env"
    stylemusic_path: Path = BASE_DIR / "stylemusic.json"
    settings_path: Path = BASE_DIR / "settings.txt"
    debug_log: Path = BASE_DIR / "webui_debug.log"
    # Реальная папка ComfyUI-воркфлоу в этом репозитории — "Workflow" (с большой буквы!)
    workflow_dir: Path = BASE_DIR / "Workflow"
    # Каталог input того ComfyUI, куда экспортируются картинки/промпты.
    # На другой машине задай: set COMFY_INPUT_DIR=D:\ComfyUiVid\input
    comfy_input_dir: Path = field(
        default_factory=lambda: Path(_env("COMFY_INPUT_DIR", r"D:\ComfyUiVid\input"))
    )


PATHS = Paths()


# ──────────────────────────────────────────────
# СЕТЬ: порты, URL, таймауты, ретраи
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class NetConfig:
    kobold_host: str = _env("KOBOLD_HOST", "127.0.0.1")
    kobold_port: int = int(_env("KOBOLD_PORT", "5001"))
    llama_host: str = _env("LLAMA_HOST", "127.0.0.1")
    llama_port: int = int(_env("LLAMA_PORT", "8080"))
    mtp_port: int = int(_env("MTP_PORT", "8510"))
    remote_llama: str = _env("REMOTE_LLAMA", "192.168.181.93:7860")
    webui_port: int = int(_env("WEBUI_PORT", "7860"))

    health_timeout: float = float(_env("HEALTH_TIMEOUT", "5"))
    cloud_timeout: float = float(_env("CLOUD_TIMEOUT", "120"))
    local_timeout: float = float(_env("LOCAL_TIMEOUT", "600"))

    # Повторные попытки для сетевых запросов.
    # Единый бюджет повторов применяется только внешним циклом http_post/http_get
    # (соединения, таймауты и 429/5xx); транспортный urllib3-Retry отключён
    # (см. rag_net.make_session), чтобы слои повторов не перемножались.
    max_retries: int = int(_env("HTTP_RETRIES", "3"))
    backoff_base: float = float(_env("HTTP_BACKOFF_BASE", "1.0"))  # сек; 1→2→4

    @property
    def kobold_url(self) -> str:
        """Return the chat-completions URL for the configured KoboldCpp host and port."""
        return f"http://{self.kobold_host}:{self.kobold_port}/v1/chat/completions"

    @property
    def kobold_info_url(self) -> str:
        """Return the info URL used to check the configured KoboldCpp server."""
        return f"http://{self.kobold_host}:{self.kobold_port}/api/v1/info"

    @property
    def llama_url(self) -> str:
        """Return the chat-completions URL for the primary llama host and port."""
        return f"{_scheme(self.llama_host)}://{self.llama_host}:{self.llama_port}/v1/chat/completions"

    @property
    def llama_endpoints(self) -> list[tuple[str, str]]:
        """Пары (health-URL, chat-URL) в порядке приоритета.

        Loopback-хосты — http://; не-локальные (удалённый ПК/сервер) — https://,
        чтобы сообщения чата передавались по шифрованному соединению.
        """
        l = self.llama_host
        s = _scheme(l)
        pairs = [
            (f"{s}://{l}:{self.llama_port}/health",
             f"{s}://{l}:{self.llama_port}/v1/chat/completions"),
            (f"{s}://{l}:{self.mtp_port}/health",
             f"{s}://{l}:{self.mtp_port}/v1/chat/completions"),
        ]
        if self.remote_llama:
            rs = _scheme(self.remote_llama)
            pairs.append((f"{rs}://{self.remote_llama}/health",
                          f"{rs}://{self.remote_llama}/v1/chat/completions"))
        return pairs


NET = NetConfig()


# ──────────────────────────────────────────────
# RAG-параметры
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class RagConfig:
    top_k: int = int(_env("RAG_TOP_K", "7"))
    collection: str = _env("RAG_COLLECTION", "zaebalo")
    default_max_tokens: int = int(_env("RAG_MAX_TOKENS", "6000"))
    default_temperature: float = float(_env("RAG_TEMPERATURE", "0.75"))


RAG = RagConfig()


# ──────────────────────────────────────────────
# ОБЛАЧНЫЕ ПРОВАЙДЕРЫ (именование ключей унифицировано: <PROVIDER>_API_KEY)
# ──────────────────────────────────────────────

CLOUD_PROVIDERS: dict[str, dict[str, str]] = {
    "OpenAI":     {"url": "https://api.openai.com/v1/chat/completions",              "model": "gpt-4o-mini",                "key_var": "OPENAI_API_KEY"},
    "Groq":       {"url": "https://api.groq.com/openai/v1/chat/completions",         "model": "llama-3.3-70b-versatile",    "key_var": "GROQ_API_KEY"},
    "DeepSeek":   {"url": "https://api.deepseek.com/chat/completions",               "model": "deepseek-chat",              "key_var": "DEEPSEEK_API_KEY"},
    "NVIDIA_NIM": {"url": "https://integrate.api.nvidia.com/v1/chat/completions",    "model": "meta/llama-3.3-70b-instruct","key_var": "NVIDIA_NIM_API_KEY"},
    "OpenRouter": {"url": "https://openrouter.ai/api/v1/chat/completions",           "model": "openai/gpt-4o-mini",         "key_var": "OPENROUTER_API_KEY"},
    "Google":     {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "model": "gemini-2.0-flash", "key_var": "GOOGLE_API_KEY"},
    "Mistral":    {"url": "https://api.mistral.ai/v1/chat/completions",              "model": "mistral-large-latest",       "key_var": "MISTRAL_API_KEY"},
    "Odirouter":  {"url": "https://api.odirouter.ai/v1/chat/completions",            "model": "grok-4.5",                   "key_var": "ODIROUTER_API_KEY"},
    "Agnes":      {"url": "https://apihub.agnes-ai.com/v1/chat/completions",         "model": "agnes-2.5-flash",            "key_var": "AGNES_API_KEY"},
}

# Совместимость со старым написанием в .env (agnes_API_KEY → AGNES_API_KEY)
KEY_ALIASES: dict[str, str] = {
    "AGNES_API_KEY": ["agnes_API_KEY"],
}
