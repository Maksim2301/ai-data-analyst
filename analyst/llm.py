import os
import time
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
RETRYABLE_CODES = {429, 500, 502, 503, 504}
_client = None

def get_client():
    global _client
    if _client is None:
        from google import genai

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Не знайдено GEMINI_API_KEY. Додайте його у файл .env")
        _client = genai.Client(api_key=key)
    return _client

def _is_transient(e: Exception) -> bool:
    code = getattr(e, "code", None)
    if code is not None:
        return code in RETRYABLE_CODES
    return any(k in type(e).__name__ for k in ("Timeout", "Connect", "Transport", "Network"))

def _call(prompt, config, retries=4):
    last = None
    for attempt in range(retries):
        try:
            return get_client().models.generate_content(model=MODEL, contents=prompt, config=config)
        except Exception as e:
            if not _is_transient(e):  # напр. 400/401/403/404 — повтор не допоможе
                raise RuntimeError(f"Gemini API відхилив запит: {e}") from e
            last = e
            time.sleep(2 ** attempt * 3)  # 3, 6, 12, 24 секунди
    raise RuntimeError(f"Gemini API недоступний після {retries} спроб: {last}")

def _text_or_raise(resp) -> str:
    if resp.text:
        return resp.text
    reason = None
    try:
        reason = resp.candidates[0].finish_reason
    except Exception:
        pass
    raise RuntimeError(
        f"Gemini повернув порожню відповідь (finish_reason={reason}). "
        "Спробуйте переформулювати запит або збільшити max_output_tokens."
    )

def ask_text(prompt, system=None, temperature=0.3):
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, max_output_tokens=8192
    )
    return _text_or_raise(_call(prompt, config))


def ask_json(prompt, schema, system=None, temperature=0.2):
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=8192,
    )
    resp = _call(prompt, config)
    parsed = getattr(resp, "parsed", None)
    if parsed is not None and isinstance(parsed, schema):
        return parsed
    return schema.model_validate_json(_text_or_raise(resp))
