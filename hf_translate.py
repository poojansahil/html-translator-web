"""
HuggingFace Inference API wrapper for English → Telugu translation.
Model: facebook/nllb-200-distilled-600M (free, no GPU needed, good quality)
"""
import os
import time
import requests

HF_API_URL = 'https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M'
MAX_CHARS = 500   # safe limit for this model


def _get_headers():
    token = os.getenv('HF_TOKEN', '')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def _translate_chunk(text: str, retries: int = 3) -> str:
    payload = {
        'inputs': text,
        'parameters': {
            'src_lang': 'eng_Latn',
            'tgt_lang': 'tel_Telu',
        }
    }
    for attempt in range(retries):
        resp = requests.post(HF_API_URL, json=payload, headers=_get_headers(), timeout=60)
        if resp.status_code == 503:
            # Model loading — wait and retry
            wait = resp.json().get('estimated_time', 20)
            time.sleep(min(wait, 30))
            continue
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return result[0].get('translation_text', text)
        return text
    raise RuntimeError(f'Translation failed after {retries} retries')


def translate_text(text: str) -> str:
    """Translate a single English string to Telugu."""
    if len(text) <= MAX_CHARS:
        return _translate_chunk(text)

    # Split long text at sentence boundaries
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
    """Translate a list of strings, calling progress_cb(done, total) after each."""
    results = []
    for i, text in enumerate(texts):
        results.append(translate_text(text))
        if progress_cb:
            progress_cb(i + 1, len(texts))
        time.sleep(0.15)   # gentle rate limiting
    return results
