import sys
sys.path.insert(0, '.')
from src.ingestion.extractor import extract_pdf

pdf_path = 'data/real_corpus/291NT8DE5483CA3F84609A0CA56720CD64103.pdf'
doc = extract_pdf(pdf_path, '291NT8DE5483CA3F84609A0CA56720CD64103.pdf')
print('Pages attribute type:', type(doc.pages))
print('Number of pages:', len(doc.pages))
for i, page in enumerate(doc.pages[:3]):
    print(f'--- Page {i+1} ---')
    print(f'Type: {type(page)}')
    print(f'Content: {repr(page[:200]) if hasattr(page, "__getitem__") else page}')
    if hasattr(page, '__getitem__'):
        try:
            print(f'First 200 chars: {repr(page[:200])}')
        except:
            print(f'First 200: cannot index')
    print()
PYEOF