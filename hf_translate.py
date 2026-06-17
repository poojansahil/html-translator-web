"""
HuggingFace Inference API wrapper for English → Telugu translation.
Model: facebook/nllb-200-distilled-600M
Sends texts in batches of BATCH_SIZE to minimize round-trips.
"""
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HF_API_URL = 'https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M'
MAX_CHARS = 400   # per-string limit inside a batch
BATCH_SIZE = 8    # strings per API call


def _make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['POST'],
        raise_on_status=False,
    )
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


_session = _make_session()


def _get_headers():
    token = os.getenv('HF_TOKEN', '')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def _call_api(inputs: list[str], attempt: int = 0) -> list[str]:
    """Send a batch of strings, return translated strings in same order."""
    payload = {
        'inputs': inputs,
        'parameters': {'src_lang': 'eng_Latn', 'tgt_lang': 'tel_Telu'},
    }
    try:
        resp = _session.post(HF_API_URL, json=payload, headers=_get_headers(), timeout=120)
    except requests.exceptions.ConnectionError:
        if attempt < 4:
            time.sleep(3 * (attempt + 1))
            return _call_api(inputs, attempt + 1)
        raise

    if resp.status_code == 503:
        wait = min(resp.json().get('estimated_time', 25), 45)
        time.sleep(wait)
        return _call_api(inputs, attempt)

    if resp.status_code == 401:
        raise RuntimeError('Invalid HF_TOKEN — check the env var in Render')

    resp.raise_for_status()
    result = resp.json()

    # Response is a list of dicts when input is a list
    if isinstance(result, list):
        out = []
        for item, original in zip(result, inputs):
            if isinstance(item, dict):
                out.append(item.get('translation_text', original))
            elif isinstance(item, list) and item:
                out.append(item[0].get('translation_text', original))
            else:
                out.append(original)
        return out

    return inputs   # fallback: return originals unchanged


def _truncate(text: str) -> str:
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def translate_batch(texts: list[str], progress_cb=None) -> list[str]:
    """
    Translate all texts, calling progress_cb(done, total) after each batch.
    Returns translated strings in the same order as input.
    """
    results = [''] * len(texts)
    total = len(texts)
    done = 0

    # Split into batches
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    for batch in batches:
        truncated = [_truncate(t) for t in batch]
        translated = _call_api(truncated)
        for j, t in enumerate(translated):
            results[done + j] = t
        done += len(batch)
        if progress_cb:
            progress_cb(done, total)
        time.sleep(0.2)   # gentle rate limiting between batches

    return results


def translate_text(text: str) -> str:
    """Translate a single string (used by manual apply path)."""
    return _call_api([_truncate(text)])[0]
