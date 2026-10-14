import sys
sys.path.insert(0, '.')
from src.ingestion.extractor import extract_pdf

pdf_path = 'data/real_corpus/291NT8DE5483CA3F84609A0CA56720CD64103.pdf'
doc = extract_pdf(pdf_path, '291NT8DE5483CA3F84609A0CA56720CD64103.pdf')
print('Attributes:', [a for a in dir(doc) if not a.startswith('__')])
print()
print('text:', repr(doc.text[:200]) if hasattr(doc, 'text') else 'NO text attr')
print('full_text:', repr(doc.full_text[:200]) if hasattr(doc, 'full_text') else 'NO full_text attr')
print('page_count:', doc.page_count)
print('total_chars:', doc.total_chars)
print('errors:', doc.errors)
print('warnings:', doc.warnings)
print('is_usable:', doc.is_usable)
print('ocr_pages:', doc.ocr_pages)
print('chunk_count:', doc.chunk_count if hasattr(doc, 'chunk_count') else 'N/A')