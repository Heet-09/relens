"""Single provider wrapper: Gemini for chat/structured-output/vision, local
sentence-transformers for embeddings.

Every other module talks to the LLM only through this file, so the rest of the
codebase doesn't care which SDK or model version is in use.
"""

import logging
import os
import time
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

T = TypeVar("T", bound=BaseModel)

_gemini_client: genai.Client | None = None
_embedder: SentenceTransformer | None = None


def _gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


def _embedding_model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info("Loading local embedding model %s (first run downloads it)...", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model ready.")
    return _embedder


def _generate_content(**kwargs):
    """generate_content with a short retry for transient 503 (model overloaded) errors."""
    label = kwargs.get("model", GEMINI_MODEL)
    for attempt in range(MAX_RETRIES):
        try:
            resp = _gemini().models.generate_content(**kwargs)
            log.debug("Gemini call ok (model=%s, attempt=%d)", label, attempt + 1)
            return resp
        except genai_errors.ServerError as exc:
            if attempt == MAX_RETRIES - 1:
                log.error("Gemini call failed after %d attempts: %s", MAX_RETRIES, exc)
                raise
            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
            log.warning(
                "Gemini overloaded (attempt %d/%d), retrying in %ds: %s", attempt + 1, MAX_RETRIES, wait, exc
            )
            time.sleep(wait)


def chat_text(prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
    log.debug("chat_text: prompt_len=%d max_tokens=%d", len(prompt), max_tokens)
    config = genai_types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
    resp = _generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
    text = (resp.text or "").strip()
    log.debug("chat_text: response_len=%d", len(text))
    return text


def generate_structured(prompt: str, system: str, schema: type[T], max_tokens: int = 2048) -> T:
    """Force Gemini to return JSON matching `schema`, parsed into an instance of it."""
    log.debug("generate_structured: schema=%s prompt_len=%d max_tokens=%d", schema.__name__, len(prompt), max_tokens)
    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = _generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
    try:
        result = schema.model_validate_json(resp.text)
    except Exception:
        log.error("generate_structured: failed to parse %s from response (len=%s)", schema.__name__, len(resp.text or ""))
        raise
    log.debug("generate_structured: parsed %s ok", schema.__name__)
    return result


def vision_extract(image_bytes: bytes, prompt: str) -> str:
    """Read a rasterized brochure page via Gemini vision (used for scanned/low-text pages)."""
    log.debug("vision_extract: image_bytes=%d", len(image_bytes))
    resp = _generate_content(
        model=GEMINI_MODEL,
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )
    return (resp.text or "").strip()


def embed(texts: list[str]) -> list[list[float]]:
    """Local embeddings via sentence-transformers — no API key, no data leaves the machine."""
    if not texts:
        return []
    log.debug("embed: %d text(s)", len(texts))
    vectors = _embedding_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()
