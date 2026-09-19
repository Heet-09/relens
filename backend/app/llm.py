"""Single provider wrapper: Gemini for chat/structured-output/vision, local
sentence-transformers for embeddings.

Every other module talks to the LLM only through this file, so the rest of the
codebase doesn't care which SDK or model version is in use.
"""

import logging
import os
import time
from typing import TypeVar

import httpx
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
REQUEST_TIMEOUT_MS = 60_000  # without this, a stalled call hangs forever — nothing times it out

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
    """generate_content with a bounded timeout (so a stalled call fails instead of hanging
    forever) and a short retry for transient errors: 503 overload, 429 rate-limit, and
    network timeouts/drops. Other 4xx errors (bad request, auth, ...) are not retried."""
    label = kwargs.get("model", GEMINI_MODEL)
    config = kwargs.get("config") or genai_types.GenerateContentConfig()
    config.http_options = genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
    kwargs["config"] = config

    for attempt in range(MAX_RETRIES):
        try:
            resp = _gemini().models.generate_content(**kwargs)
            log.debug("Gemini call ok (model=%s, attempt=%d)", label, attempt + 1)
            return resp
        except genai_errors.ClientError as exc:
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
            log.warning(
                "Gemini rate-limited (attempt %d/%d), retrying in %ds: %s", attempt + 1, MAX_RETRIES, wait, exc
            )
            time.sleep(wait)
        except (genai_errors.ServerError, httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt == MAX_RETRIES - 1:
                log.error("Gemini call failed after %d attempts: %s", MAX_RETRIES, exc)
                raise
            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
            log.warning(
                "Gemini call failed (attempt %d/%d), retrying in %ds: %s", attempt + 1, MAX_RETRIES, wait, exc
            )
            time.sleep(wait)


def chat_text(prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
    log.debug("chat_text: prompt_len=%d max_tokens=%d", len(prompt), max_tokens)
    config = genai_types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
    resp = _generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
    text = (resp.text or "").strip()
    log.debug("chat_text: response_len=%d", len(text))
    return text


STRUCTURED_PARSE_RETRIES = 3
REPETITION_MIN_WORDS = 20  # below this, a low unique-word ratio is just a short natural phrase
REPETITION_UNIQUE_RATIO = 0.3  # below this fraction of unique words, treat as a decoding loop


def _repetition_loop_field(obj) -> str | None:
    """Gemini occasionally falls into a decoding loop on a long free-text field (e.g. `summary`)
    and fills it with the same handful of filler words repeated dozens of times — valid JSON, so
    the parse succeeds, but the content is garbage. Returns the offending text if found, so the
    caller can retry the same way it retries genuinely invalid JSON."""
    texts: list[str] = []

    def collect(o):
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)

    collect(obj)
    for text in texts:
        words = text.split()
        if len(words) < REPETITION_MIN_WORDS:
            continue
        if len(set(words)) / len(words) < REPETITION_UNIQUE_RATIO:
            return text
    return None


def generate_structured(prompt: str, system: str, schema: type[T], max_tokens: int = 2048) -> T:
    """Force Gemini to return JSON matching `schema`, parsed into an instance of it. Retried a
    couple times on invalid JSON — occasionally the model degenerates into a repetition loop and
    gets cut off mid-object by max_output_tokens; re-asking usually gets a clean, shorter response.
    Also retried if parsing succeeds but a text field is itself a repetition-loop (valid JSON,
    garbage content) — see `_repetition_loop_field`."""
    log.debug("generate_structured: schema=%s prompt_len=%d max_tokens=%d", schema.__name__, len(prompt), max_tokens)
    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=schema,
    )
    last_exc: Exception | None = None
    for attempt in range(STRUCTURED_PARSE_RETRIES):
        resp = _generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
        try:
            result = schema.model_validate_json(resp.text)
        except Exception as exc:
            last_exc = exc
            log.warning(
                "generate_structured: invalid JSON for %s (attempt %d/%d, response_len=%s): %s",
                schema.__name__,
                attempt + 1,
                STRUCTURED_PARSE_RETRIES,
                len(resp.text or ""),
                exc,
            )
            continue

        looped_text = _repetition_loop_field(result.model_dump())
        if looped_text is not None:
            last_exc = ValueError(f"repetition-loop degeneracy detected in a text field: {looped_text[:200]!r}...")
            log.warning(
                "generate_structured: repetition-loop degeneracy for %s (attempt %d/%d): %s",
                schema.__name__,
                attempt + 1,
                STRUCTURED_PARSE_RETRIES,
                last_exc,
            )
            continue

        log.debug("generate_structured: parsed %s ok (attempt %d)", schema.__name__, attempt + 1)
        return result
    log.error("generate_structured: failed to parse %s after %d attempt(s)", schema.__name__, STRUCTURED_PARSE_RETRIES)
    raise last_exc


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
