import sys
sys.path.insert(0, '.')
from src.ingestion.extractor import extract_pdf

pdfs = [
    'data/real_corpus/2610.13644v1.pdf',
    'data/real_corpus/291NT8DE5483CA3F84609A0CA56720CD64103.pdf',
    'data/real_corpus/292NT63009643AFA847A48C9A73F31E26175F.pdf',
]

for pdf_path in pdfs:
    name = pdf_path.rsplit('\\', 1)[-1]
    try:
        doc = extract_pdf(pdf_path, name)
        print(f'=== {name} ===')
        print(f'  Pages: {doc.page_count}')
        print(f'  Total chars: {doc.total_chars}')
        print(f'  Errors: {doc.errors}')
        print(f'  Warnings: {doc.warnings}')
        print(f'  Is usable: {doc.is_usable}')
        print(f'  OCR pages: {doc.ocr_pages}')
        preview = doc.text[:500].replace('\n', ' ')
        print(f'  Preview: {preview}...')
        print()
    except Exception as e:
        print(f'=== {name} ===')
        print(f'  Error: {e}')
        print()