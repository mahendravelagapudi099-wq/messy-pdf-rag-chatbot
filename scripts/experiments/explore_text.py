import sys
sys.path.insert(0, '.')
from src.ingestion.extractor import extract_pdf

pdf_path = 'data/real_corpus/291NT8DE5483CA3F84609A0CA56720CD64103.pdf'
doc = extract_pdf(pdf_path, '291NT8DE5483CA3F84609A0CA56720CD64103.pdf')
print('Total chars:', doc.total_chars)
print('First 300 chars from extracted text:')
# Try to get text from pages
for i, page in enumerate(doc.pages):
    try:
        txt = page.text if hasattr(page, 'text') else str(page)
        print(f'Page {i+1} first 150: {txt[:150]}')
    except Exception as e:
        print(f'Page {i+1} error: {e}')
print()
print('doc_id:', doc.doc_id)
print('filename:', doc.filename)
print('page_count:', doc.page_count)
PYEOF