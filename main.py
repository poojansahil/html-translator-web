import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from translator import (extract_to_excel, load_translations_from_bytes,
                        apply_translations, walk_dom, strip_emoji)
from hf_translate import translate_batch
from bs4 import BeautifulSoup

app = FastAPI(title='HTML Translator')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/extract')
async def extract(file: UploadFile = File(...)):
    """Upload HTML → download Excel template."""
    if not file.filename.endswith('.html'):
        raise HTTPException(400, 'Please upload an .html file')
    html = (await file.read()).decode('utf-8', errors='replace')
    xlsx_bytes = extract_to_excel(html)
    stem = Path(file.filename).stem
    return Response(
        content=xlsx_bytes,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename="{stem}_translations.xlsx"'}
    )


@app.post('/auto-translate')
async def auto_translate(file: UploadFile = File(...)):
    """
    Upload HTML → stream SSE progress events → final event contains
    a data-URL of the Telugu HTML so the browser can download it.
    """
    if not file.filename.endswith('.html'):
        raise HTTPException(400, 'Please upload an .html file')
    html = (await file.read()).decode('utf-8', errors='replace')
    stem = Path(file.filename).stem

    soup = BeautifulSoup(html, 'html.parser')
    records = list(walk_dom(soup))
    if not records:
        raise HTTPException(400, 'No translatable text found in this HTML file')

    texts = [strip_emoji(text) for _, _, _, text in records]
    total = len(texts)

    async def event_stream():
        translated_results = {}
        error_holder = {}

        # Progress callback runs in the thread — queues events via asyncio
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def progress_cb(done, total):
            pct = int(done / total * 100)
            loop.call_soon_threadsafe(queue.put_nowait, ('progress', done, total, pct))

        def run_translation():
            try:
                results = translate_batch(texts, progress_cb=progress_cb)
                loop.call_soon_threadsafe(queue.put_nowait, ('done', results))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ('error', str(e)))

        # Start translation in background thread
        asyncio.get_event_loop().run_in_executor(None, run_translation)

        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        while True:
            msg = await queue.get()
            if msg[0] == 'progress':
                _, done, tot, pct = msg
                yield f"data: {json.dumps({'type': 'progress', 'done': done, 'total': tot, 'pct': pct})}\n\n"
            elif msg[0] == 'done':
                translated = msg[1]
                # Build translations dict and apply
                translations = {}
                english_check = {}
                for (idx, _, _, original), telugu in zip(records, translated):
                    translations[idx] = telugu
                    english_check[idx] = original
                new_html, applied, _ = apply_translations(html, translations, english_check)
                yield f"data: {json.dumps({'type': 'complete', 'html': new_html, 'filename': stem + '_Telugu.html'})}\n\n"
                break
            elif msg[0] == 'error':
                yield f"data: {json.dumps({'type': 'error', 'message': msg[1]})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.post('/apply')
async def apply(
    html_file: UploadFile = File(...),
    excel_file: UploadFile = File(...)
):
    """Upload original HTML + filled Excel → download Telugu HTML."""
    if not html_file.filename.endswith('.html'):
        raise HTTPException(400, 'First file must be .html')
    if not excel_file.filename.endswith('.xlsx'):
        raise HTTPException(400, 'Second file must be .xlsx')

    html = (await html_file.read()).decode('utf-8', errors='replace')
    xlsx_bytes = await excel_file.read()

    try:
        translations, english_check = load_translations_from_bytes(xlsx_bytes)
    except Exception as e:
        raise HTTPException(400, f'Could not read Excel file: {e}')

    if not translations:
        raise HTTPException(400, 'No translations found in the Excel file — fill column F first')

    new_html, applied, total = apply_translations(html, translations, english_check)
    stem = Path(html_file.filename).stem
    return Response(
        content=new_html.encode('utf-8'),
        media_type='text/html; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename="{stem}_Telugu.html"'}
    )


# Serve the frontend
static_dir = Path(__file__).parent / 'static'
app.mount('/', StaticFiles(directory=str(static_dir), html=True), name='static')
