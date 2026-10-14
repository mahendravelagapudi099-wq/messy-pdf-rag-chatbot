import sys
sys.path.insert(0, '.')
from src.ingestion.extractor import extract_pdf

pdf_path = 'data/real_corpus/291NT8DE5483CA3F84609A0CA56720CD64103.pdf'
doc = extract_pdf(pdf_path, '291NT8DE5483CA3F84609A0CA56720CD64103.pdf')
print('Pages attribute type:', type(doc.pages))
print('Number of pages:', len(doc.pages))
for i, page in enumerate(doc.pages):
    print(f'--- Page {i+1} ---')
    print(f'Page repr: {repr(page)[:300]}')
    print()
PYEOF