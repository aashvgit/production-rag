import os
import time

from google import genai

MODEL = os.environ.get("GEMINI_MODEL", "")
_client = None


def generate(prompt: str, retries: int = 5) -> str:
    if not MODEL:
        raise RuntimeError("Set the GEMINI_MODEL environment variable first.")
    global _client
    if _client is None:
        _client = genai.Client()
    for attempt in range(retries):
        try:
            response = _client.models.generate_content(model=MODEL, contents=prompt)
            return response.text or ""
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(5 * 2**attempt)  # wait 5s, 10s, 20s, ...
                continue
            raise
    raise RuntimeError("Still rate limited after retries. Wait a while or slow down.")