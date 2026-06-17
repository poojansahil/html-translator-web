import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from translator import extract_to_excel, load_translations_from_bytes, apply_translations, walk_dom, strip_emoji, should_extract
from hf_translate import translate_text, translate_batch
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
    """Upload an HTML file → download a filled Excel template."""
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
    """Upload an HTML file → auto-translate to Telugu → download Telugu HTML."""
    if not file.filename.endswith('.html'):
        raise HTTPException(400, 'Please upload an .html file')
    html = (await file.read()).decode('utf-8', errors='replace')

    # Extract all English strings
    soup = BeautifulSoup(html, 'html.parser')
    records = list(walk_dom(soup))
    if not records:
        raise HTTPException(400, 'No translatable text found in this HTML file')

    # Translate each string
    texts_to_translate = [strip_emoji(text) for _, _, _, text in records]
    try:
        translated = await asyncio.to_thread(translate_batch, texts_to_translate)
    except Exception as e:
        raise HTTPException(502, f'Translation service error: {e}')

    # Build translations dict (index → telugu) and english_check (index → original)
    translations = {}
    english_check = {}
    for (idx, _, _, original), telugu in zip(records, translated):
        translations[idx] = telugu
        english_check[idx] = original

    new_html, applied, total = apply_translations(html, translations, english_check)
    stem = Path(file.filename).stem
    return Response(
        content=new_html.encode('utf-8'),
        media_type='text/html; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename="{stem}_Telugu.html"'}
    )


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
