"""
HuggingFace Inference API wrapper for English → Telugu translation.
Model: facebook/nllb-200-distilled-600M (free, no GPU needed, good quality)
"""
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HF_API_URL = 'https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M'
MAX_CHARS = 500


def _make_session():
    """Requests session with automatic retry on connection errors."""
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


def _translate_chunk(text: str, retries: int = 4) -> str:
    payload = {
        'inputs': text,
        'parameters': {
            'src_lang': 'eng_Latn',
            'tgt_lang': 'tel_Telu',
        }
    }
    for attempt in range(retries):
        try:
            resp = _session.post(HF_API_URL, json=payload, headers=_get_headers(), timeout=90)
        except requests.exceptions.ConnectionError as e:
            # DNS or connection failure — wait and retry
            wait = 3 * (attempt + 1)
            time.sleep(wait)
            continue

        if resp.status_code == 503:
            wait = min(resp.json().get('estimated_time', 20), 40)
            time.sleep(wait)
            continue

        if resp.status_code == 401:
            raise RuntimeError('Invalid HF_TOKEN — set a valid token in Render env vars')

        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return result[0].get('translation_text', text)
        return text

    raise RuntimeError(f'Translation failed after {retries} attempts')


def translate_text(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return _translate_chunk(text)

    sentences = [s.strip() for s in text.replace('\n', '. ').split('. ') if s.strip()]
    chunks, current = [], ''
    for s in sentences:
        if len(current) + len(s) + 2 <= MAX_CHARS:
            current = (current + '. ' + s).strip('. ')
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    parts = []
    for chunk in chunks:
        parts.append(_translate_chunk(chunk))
        time.sleep(0.1)
    return '. '.join(parts)


def translate_batch(texts: list[str], progress_cb=None) -> list[str]:
    results = []
    for i, text in enumerate(texts):
        results.append(translate_text(text))
        if progress_cb:
            progress_cb(i + 1, len(texts))
        time.sleep(0.15)
    return results
