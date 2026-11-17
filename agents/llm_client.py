"""
FairSplit AI — Provider-Agnostic LLM Client

Supports 3 providers (set via LLM_PROVIDER env var):
  - 'gemini'     : Google Gemini (DEFAULT — free tier, needs GOOGLE_API_KEY)
  - 'anthropic'  : Claude (needs ANTHROPIC_API_KEY)
  - 'openai'     : GPT-4o (needs OPENAI_API_KEY)

If LLM_PROVIDER is not set, auto-detects based on available keys.
Priority: GOOGLE_API_KEY → ANTHROPIC_API_KEY → OPENAI_API_KEY
"""

import os
import time
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Streamlit Cloud secrets support
# On Streamlit Cloud, .env is not available (gitignored). Instead, secrets
# are configured through the Streamlit Cloud dashboard and accessed via
# st.secrets. We inject them into os.environ so the rest of the code works
# transparently.
# ---------------------------------------------------------------------------
try:
    import streamlit as st

    for key in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        if key not in os.environ and key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # Not running in Streamlit, or st.secrets not configured

logger = logging.getLogger(__name__)

_llm_instance = None


def _detect_provider() -> str:
    """
    Auto-detect the best available provider based on which API keys are set.
    Priority: gemini (free) → anthropic → openai.
    """
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Default to gemini — the user likely forgot to set the key
    return "gemini"


def get_llm():
    """
    Return a LangChain chat model based on the LLM_PROVIDER env var.
    If not set, auto-detects the best available provider.

    Supported providers:
      - 'gemini'    (default, free tier available)
      - 'anthropic' (Claude)
      - 'openai'    (GPT-4o)
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        provider = _detect_provider()
        logger.info("Auto-detected LLM provider: %s", provider)

    # --- Google Gemini (free tier) ---
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. Get a FREE key at https://aistudio.google.com/apikey\n"
                "Then add it to your .env file:\n"
                "  GOOGLE_API_KEY=your-key-here"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.4,
            max_output_tokens=1024,
        )
        logger.info("Using Google Gemini (gemini-2.5-flash, free tier) as LLM provider.")
        return _llm_instance

    # --- Anthropic (Claude) ---
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — falling back to auto-detect."
            )
            provider = _detect_provider()
            if provider == "anthropic":
                raise EnvironmentError("ANTHROPIC_API_KEY not set.")
            # Recurse with detected provider
            os.environ["LLM_PROVIDER"] = provider
            return get_llm()

        from langchain_anthropic import ChatAnthropic

        _llm_instance = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=api_key,
            temperature=0.4,
            max_tokens=1024,
        )
        logger.info("Using Anthropic (Claude) as LLM provider.")
        return _llm_instance

    # --- OpenAI (GPT-4o) ---
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Set it in your .env file."
            )
        from langchain_openai import ChatOpenAI

        _llm_instance = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.4,
            max_tokens=1024,
        )
        logger.info("Using OpenAI (GPT-4o) as LLM provider.")
        return _llm_instance

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. "
        f"Use 'gemini' (free), 'anthropic', or 'openai'."
    )


def reset_llm():
    """Reset the cached LLM instance (useful for testing)."""
    global _llm_instance
    _llm_instance = None


def call_llm(prompt: str, max_retries: int = 3) -> str:
    """
    Send a single human message to the configured LLM and return the response text.
    Retries with exponential backoff on transient rate-limit (429) errors.
    """
    llm = get_llm()

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            # Gemini may return content as a list of parts — join them into a string
            if isinstance(content, list):
                content = "\n".join(
                    part if isinstance(part, str) else part.get("text", str(part))
                    for part in content
                )
            return content
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                logger.warning(
                    "Rate limited (attempt %d/%d) — retrying in %ds: %s",
                    attempt + 1, max_retries + 1, wait, e,
                )
                time.sleep(wait)
            else:
                raise

