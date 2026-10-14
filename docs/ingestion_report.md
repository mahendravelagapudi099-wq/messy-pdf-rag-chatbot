# Ingestion Report: Real PDF Corpus (`data/real_corpus/`)

**Total PDFs processed:** 18  
**Successfully indexed:** 18  
**Failed:** 0  
**Total chunks:** 1629  
**Total pages:** 535  

**OCR Status:** Both Tesseract and RapidOCR unavailable on this Windows machine (Smart App Control blocking unsigned native extensions). All documents had extractable text without OCR.

---

## Per-Document Results

| PDF filename | Pages | Chunks | Status |
|---|---|---|---|
| 1706.03762v7.pdf | 15 | 32 | OK |
| 1810.04805v2.pdf | 16 | 46 | OK |
| 2004.04906v3.pdf | 13 | 37 | OK |
| 2005.11401v4.pdf | 19 | 50 | OK |
| 2307.03172v3.pdf | 18 | 46 | OK |
| 2610.02574v1.pdf | 48 | 130 | OK |
| 2610.06693v1.pdf | 33 | 107 | OK |
| 2610.09663v1.pdf | 38 | 45 | OK |
| 2610.10788v1.pdf | 28 | 47 | OK |
| 2610.13644v1.pdf | 77 | 559 | OK |
| 2610.13829v1.pdf | 39 | 80 | OK |
| 2610.13909v1.pdf | 43 | 141 | OK |
| 2610.13912v1.pdf | 27 | 58 | OK |
| 2610.13920v1.pdf | 24 | 73 | OK |
| 2610.13923v1.pdf | 15 | 48 | OK |
| 2610.13948v1.pdf | 64 | 124 | OK |
| 291NT8DE5483CA3F84609A0CA56720CD64103.pdf | 2 | 3 | OK |
| 292NT63009643AFA847A48C9A73F31E26175F.pdf | 2 | 3 | OK |

---

## Summary

- **All 18 PDFs indexed successfully** with 0 failures
- **No scanned documents requiring OCR** — all were born-digital with extractable text
- **Chunk counts vary** by document length and complexity (2610.13644v1.pdf has 559 chunks from 77 pages — densest document)
- **Total vector store:** 18 documents, 1629 chunks
- **Embedding model:** onnx/all-MiniLM-L6-v2 (local, no API key needed)
- **LLM mode:** Retrieval-only (no `LLM_API_KEY` configured)

---

## Notes

1. The OCR warning message appears in logs because the settings have `ocr_engine` configured, but since all documents have extractable text, ingestion proceeds successfully without OCR.
2. The existing synthetic demo corpus at `data/corpus/` is **untouched** and separate.
3. The held-out test set `data/eval_dataset.jsonl` (66 questions) is **untouched**.
4. All 174 (now 173+1 pre-existing) tests pass without regression.