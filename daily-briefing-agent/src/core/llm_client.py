# AI Generated Code - Start
"""
LLM client configuration.

All model and endpoint settings are centralized here.
Downstream code should call `get_client()`, `get_model(purpose)`, and
`supports_json_mode()` rather than constructing OpenAI clients directly.

Supported providers
-------------------
  wanqing (default)
    - Endpoint : WQ_BASE_URL  (default: https://wanqing-api.corp.kuaishou.com/…)
    - Auth     : WQ_API_KEY
    - JSON mode: NOT supported — model wraps JSON in ```json … ``` fences.
                 Always use `extract_json(text)` to strip fences before parsing.

  openai
    - Endpoint : OPENAI_BASE_URL  (default: https://api.openai.com/v1)
    - Auth     : OPENAI_API_KEY
    - JSON mode: supported — pass response_format={"type":"json_object"} when
                 `supports_json_mode()` returns True.

Switching providers
-------------------
  Set LLM_PROVIDER=openai (or wanqing) in your .env or shell.

Configuration priority (per field)
-----------------------------------
  1. Environment variables (.env or shell export)
  2. Defaults defined in this file

Environment variables
---------------------
  LLM_PROVIDER        — "wanqing" (default) | "openai"

  # WanQing
  WQ_API_KEY          — required when provider=wanqing
  WQ_BASE_URL         — optional, override WanQing gateway URL

  # OpenAI
  OPENAI_API_KEY      — required when provider=openai
  OPENAI_BASE_URL     — optional, override base URL (e.g. Azure, proxy)

  # Model overrides (apply to whichever provider is active)
  MODEL_DEFAULT       — fallback model for any purpose
  MODEL_PROFILE       — ProfileStore tag extraction
  MODEL_FILTER        — Filter Agents (calendar / email / news)
  MODEL_RANKING       — Ranking Agent
  MODEL_WRITING       — Writing Agent
  MODEL_VALIDATION    — Validation Agent
"""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file if present (no-op if already set in shell environment)
load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_WANQING_BASE_URL    = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints"
_WANQING_DEFAULT_MODEL = "ep-wg18ya-1778230307591344312"

_OPENAI_BASE_URL     = "https://api.openai.com/v1"
# Per-purpose OpenAI model defaults: use gpt-4o-mini for high-volume filter
# tasks and gpt-4o for reasoning-heavy ranking / writing / validation.
_OPENAI_DEFAULT_MODELS: dict[str, str] = {
    "profile":    "gpt-4o-mini",
    "filter":     "gpt-4o-mini",
    "ranking":    "gpt-4o",
    "writing":    "gpt-4o",
    "validation": "gpt-4o",
    "default":    "gpt-4o",
}

# Regex to strip markdown code fences: ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_provider() -> str:
    """Return the active LLM provider: 'wanqing' or 'openai'."""
    return os.environ.get("LLM_PROVIDER", "wanqing").lower().strip()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """
    Return a singleton OpenAI-compatible client for the active provider.
    Cached — safe to call multiple times across the codebase.

    Raises EnvironmentError if the required API key is missing.
    """
    provider = get_provider()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Add it to your .env file or export it in your shell."
            )
        base_url = os.environ.get("OPENAI_BASE_URL", _OPENAI_BASE_URL)
        return OpenAI(base_url=base_url, api_key=api_key)

    # Default: wanqing
    api_key = os.environ.get("WQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "WQ_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )
    base_url = os.environ.get("WQ_BASE_URL", _WANQING_BASE_URL)
    return OpenAI(base_url=base_url, api_key=api_key)


def get_model(purpose: str = "default") -> str:
    """
    Return the model identifier for a given purpose.

    purpose options:
      "profile"    — ProfileStore tag extraction
      "filter"     — Filter Agents (calendar / email / news)
      "ranking"    — Ranking Agent
      "writing"    — Writing Agent
      "validation" — Validation Agent
      "default"    — fallback

    Each purpose can be independently overridden via MODEL_<PURPOSE> env var,
    regardless of which provider is active.
    """
    env_key_map = {
        "profile":    "MODEL_PROFILE",
        "filter":     "MODEL_FILTER",
        "ranking":    "MODEL_RANKING",
        "writing":    "MODEL_WRITING",
        "validation": "MODEL_VALIDATION",
    }
    # 1. Per-purpose env override (works for both providers)
    env_key = env_key_map.get(purpose)
    if env_key:
        override = os.environ.get(env_key)
        if override:
            return override

    # 2. Global default override
    global_override = os.environ.get("MODEL_DEFAULT")
    if global_override:
        return global_override

    # 3. Provider-specific built-in defaults
    provider = get_provider()
    if provider == "openai":
        return _OPENAI_DEFAULT_MODELS.get(purpose, _OPENAI_DEFAULT_MODELS["default"])

    return _WANQING_DEFAULT_MODEL


def supports_json_mode() -> bool:
    """
    Return True if the active provider supports response_format={"type":"json_object"}.

    OpenAI natively supports JSON mode — the response will be bare JSON with no
    markdown fences.  WanQing does NOT support it and wraps JSON in code fences.

    Usage in agents:
        kwargs = {}
        if supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(..., **kwargs)
        text = response.choices[0].message.content
        data = json.loads(extract_json(text))   # extract_json is safe either way
    """
    return get_provider() == "openai"


def extract_json(text: str) -> str:
    """
    Strip markdown code fences from LLM output and return the raw JSON string.

    Safe to call regardless of provider:
    - WanQing wraps JSON in ```json … ``` — fences are stripped.
    - OpenAI (json_object mode) returns bare JSON — returned as-is.

    Examples:
      "```json\\n{...}\\n```"  →  "{...}"
      "{...}"                  →  "{...}"   (already clean)
    """
    if not text or not text.strip():
        raise ValueError("LLM returned empty response")

    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # No fences found — return as-is (bare JSON or OpenAI json_object mode)
    return text.strip()

# AI Generated Code - End
