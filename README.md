# PDF RAG Chatbot

A working retrieval-augmented-generation chatbot for messy, real-world PDFs —
government circulars, scheme guidelines, syllabi, and scanned documents. It
answers strictly from what it can find in the uploaded files, cites the exact
page, and says so plainly when the documents do not contain the answer.

The differentiator is the **evaluation framework**: retrieval quality is measured
reproducibly rather than asserted, and a benchmark command compares chunking
configurations against a 66-question grounded dataset.

---

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Measured results](#measured-results)
- [Tests](#tests)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Quick start

```bash
# 1. Create an environment (Python 3.11+; developed and verified on 3.14)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. Install
pip install -r requirements.txt

# 3. Generate the demo corpus (synthetic PDFs, ~55 KB total)
python scripts/make_demo_corpus.py --out data/corpus

# 4. Configure
cp .env.example .env            # then add LLM_API_KEY (optional)

# 5. Index the corpus and check status
python scripts/run_eval.py ingest --corpus data/corpus
python scripts/run_eval.py status

# 6. Launch
streamlit run app.py
```

The app works **without an LLM key**. It runs in retrieval-only mode: relevant
passages are shown with scores and citations, but no generated prose. Add
`LLM_API_KEY` to enable grounded generation.

---

## Architecture

```
app.py                      Streamlit UI (presentation only)
pytest.ini                  Pytest configuration (testpaths = tests)
docs/
  ingestion_report.md       Corpus ingestion breakdown
data/
  eval_dataset.jsonl        66-question evaluation dataset
  corpus/                   Synthetic demo PDFs (generated)
scripts/
  run_eval.py               CLI: ingest / questions / benchmark / status
  make_demo_corpus.py       Generates the synthetic demo corpus
  experiments/              Corpus analysis, scratch scripts & verification tools
src/
  config.py                 Settings from env, validation, experiment configs
  models.py                 Domain types (Page, Chunk, Answer, Citation, ...)
  rag.py                    RAGEngine facade -- the UI talks only to this
  chunking.py               Page-aware, token-aware chunker
  embeddings.py             ONNX (default) / sentence-transformers providers
  vector_store.py           Persistent ChromaDB + document lifecycle
  retrieval.py              Search, thresholding, optional reranker
  generation.py             Grounded generation + offline extractive answerer
  citations.py              Citation building and lexical verification
  ingestion/
    validator.py            Type, size, magic bytes, path safety
    extractor.py            PyMuPDF text + tables, OCR routing, header stripping
    ocr.py                  Tesseract / RapidOCR backends with capability probe
    cleaner.py              Ligatures, de-hyphenation, furniture, headings
    pipeline.py             validate -> extract -> chunk -> embed -> store
  evaluation/
    dataset.py              JSONL schema, loading, coverage report
    metrics.py              Hit Rate@k, MRR, F1, fact recall, citations, ...
    benchmark.py            The runner
    report.py               CSV / JSON / Markdown / chart export
tests/                      Unit and integration test suite (174 tests)
```

**Design decisions worth knowing:**

| Concern | Decision and why |
|---|---|
| Chunking | Token-counted with `tiktoken`, never spans pages, tables are emitted as indivisible chunks. Page-accurate citations depend on the no-page-spanning invariant. |
| Dedup | `doc_id` is the SHA-256 of the raw bytes. Re-uploading the same file is reported as `skipped`, so ingestion is idempotent. Re-indexing deletes the prior vectors first, so stale rows cannot survive. |
| Embeddings | ONNX MiniLM by default — no torch, ~79 MB, CPU-only. `sentence-transformers` is opt-in. |
| Retrieval | Cosine similarity, `similarity = 1 - distance`. The threshold is what converts a weak match into an abstention. |
| Reranking | Off by default. It is a real accuracy/latency trade-off, so it is benchmarked (`--reranker`) rather than assumed. |
| Answer safety | Evidence is fenced and the system prompt states that documents are data, never instructions. |
| Abstention | Tracked separately from *degraded* answers. A missing LLM key is degraded mode; it is not an abstention, and conflating the two corrupts the abstention metric. |

---

## Configuration

All configuration is environment-driven (`.env`). See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | *(empty)* | Enables generation. Empty = retrieval-only mode. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, OpenAI). |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Model identifier. |
| `EMBEDDING_PROVIDER` | `onnx` | `onnx` or `sentence-transformers`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Token counts. Overlap must be < size. |
| `TOP_K` | `5` | Passages retrieved per question. |
| `SIMILARITY_THRESHOLD` | `0.35` | Below this, passages are discarded. |
| `USE_RERANKER` | `false` | Cross-encoder reranking (needs extras). |
| `OCR_ENGINE` | `auto` | `auto`, `tesseract`, `rapidocr`, or `none`. |
| `TESSERACT_CMD` | *(auto-detect)* | Path to the Tesseract binary. |
| `MAX_UPLOAD_MB` | `50` | Per-file upload ceiling. |

### OCR

Scanned pages are detected automatically (a page whose text layer has almost no
alphanumeric content) and routed to OCR. Backends are probed, never assumed:

- **Tesseract** — preferred. Install it and either put it on `PATH` or set
  `TESSERACT_CMD`. Extra languages: set `OCR_LANGUAGES=eng+hin`.
- **RapidOCR** — fallback with bundled ONNX models; needs no system binary.
- **None available** — the page is *reported as skipped* and ingestion continues.
  Failures are never silent.

If OCR is unavailable, scanned pages contribute no text, and questions about
them will abstain. The Settings tab shows the detected status.

---

## Evaluation

### Dataset

`data/eval_dataset.jsonl` — **66 questions** (54 answerable, 12 unanswerable):

| Category | Count | What it tests |
|---|---|---|
| `factual` | 19 | Direct lookup of a stated fact |
| `table` | 15 | Values that live inside a bordered table |
| `multi_page` | 6 | Answers requiring evidence from several pages |
| `paraphrase` | 9 | Question wording deliberately unlike the document |
| `scanned` | 5 | Answers that require OCR |
| `unanswerable` | 12 | Must abstain; no supporting evidence exists |

Each record:

```json
{"id": "table-rbi-001", "category": "table",
 "question": "In Table 1, what measure applies to high risk customers...?",
 "reference_answer": "Politically exposed persons must have their address...",
 "expected_source": "rbi_due_diligence_circular.pdf", "expected_pages": [2],
 "required_facts": ["Politically exposed persons", "Every year"],
 "should_abstain": false}
```

The ground truth is derived from the synthetic demo corpus in `data/corpus/`,
which is generated by `scripts/make_demo_corpus.py`. To build a dataset from
your own real documents, drop PDFs into a corpus directory, then write one
JSONL line per question using the schema above.

### Metric definitions

- **Retrieval Hit Rate@k** — fraction of answerable questions where an expected
  source appears in the top-k. A hit requires the expected document *and* a page
  within ±1 of an expected page, **or** a passage sharing ≥35 % of the reference
  answer's content tokens.
- **MRR** — mean of `1 / rank` of the first correct result; 0 when absent.
- **Answer correctness (token F1)** — F1 over content tokens (stopwords removed)
  between answer and reference. Deterministic, free, and deliberately **not** an
  LLM judge.
- **Required-fact recall** — fraction of `required_facts` present in the answer.
  This is what catches a fluent-but-wrong answer.
- **Citation correctness** — fraction of cited sources pointing at the expected
  document and page.
- **Unsupported-answer rate** — among non-abstained answers to answerable
  questions, the fraction with fact recall < 0.5. A deterministic hallucination
  proxy.
- **Appropriate abstention rate** — among `should_abstain` questions, the
  fraction where the system abstained.
- **Ingestion failure rate** — documents that yielded no usable text.

### Running it

```bash
# Dataset sanity check (no model needed)
python scripts/run_eval.py questions --dataset data/eval_dataset.jsonl --list

# Chunking comparison across 256/32, 512/64, 1024/128 (no API key needed)
python scripts/run_eval.py benchmark --corpus data/corpus

# Single configuration, with a real LLM
python scripts/run_eval.py benchmark --corpus data/corpus --mode llm --chunk-size 512

# Also test the cross-encoder reranker
python scripts/run_eval.py benchmark --corpus data/corpus --reranker
```

Artifacts are written to `results/` with a UTC run stamp:

| File | Contents |
|---|---|
| `run-<ts>-compare.csv` | One row per configuration |
| `run-<ts>-compare_summary.json` | Metrics + dataset coverage |
| `run-<ts>-cs<N>.json` | Full per-question outcomes (for error analysis) |
| `run-<ts>-compare_report.md` | Human-readable report |
| `run-<ts>-compare_chart.png` | Grouped bar chart |

### Two answer modes

`--mode extractive` (default) is deterministic and needs no API key: it selects
the retrieved sentences most relevant to the question. This makes retrieval
quality measurable on any machine and in CI. It is **not** a generative baseline
— `--mode llm` is required to evaluate the generation path.

---

## Measured results

Produced by `python scripts/run_eval.py benchmark --corpus data/corpus` on
Python 3.14 / Windows, ONNX MiniLM embeddings, top-k = 5, threshold = 0.0,
**extractive** answer mode, 66 questions.

| config | Hit Rate@5 | MRR | Answer F1 | Fact recall | Citation correct | Unsupported | Abstention | Latency |
|---|---|---|---|---|---|---|---|---|
| 256 / 32 | 0.926 | 0.893 | 0.201 | 0.654 | 0.815 | 0.255 | 0.250 | 0.141 s |
| **512 / 64** | **0.926** | **0.893** | **0.207** | **0.673** | **0.833** | **0.250** | **0.250** | **0.058 s** |
| 1024 / 128 | 0.926 | 0.893 | 0.207 | 0.673 | 0.833 | 0.250 | 0.250 | 0.062 s |

Ingestion: 3 of 4 documents indexed; 1 failed (the scanned PDF — no OCR backend
available on this machine). That failure is reported, not swallowed.

**Reading these numbers honestly:**

1. **Chunk size barely matters here, and that is a property of the corpus.**
   Hit rate and MRR are identical across all three configurations because the
   demo documents are short: each page already fits in one chunk, so 256 vs 1024
   tokens changes nothing. This benchmark does **not** demonstrate that chunk
   size is irrelevant on real 200-page circulars. Re-run it against a realistic
   corpus before drawing that conclusion. 512/64 is chosen as the default on the
   basis of latency alone.

2. **Answer F1 ≈ 0.2 is expected and not a failure signal.** The extractive
   answerer emits 1–3 sentences against multi-sentence references, so token
   overlap is structurally low. This number measures the extractive baseline, not
   the LLM path.

3. **Abstention rate of 0.25 is the weakest result.** Only 3 of 12 unanswerable
   questions were correctly declined. The extractive answerer will fall back on a
   sentence sharing a few content words. Raising the bar would raise this number
   but would be tuning on the evaluation set, so the threshold was left alone and
   the weakness is reported instead.

4. **The `scanned` questions are expected to fail without OCR** and they do. They
   exist to make that gap visible rather than to flatter the numbers.

No baseline was invented and no result is claimed that was not produced by an
actual run.

---

## Tests

```bash
pytest -q
# or: python -m pytest -q
```

**174 tests, all passing.** Coverage by area:

| File | What it covers |
|---|---|
| `test_validator.py` | Magic-byte sniffing, renamed-file rejection, path traversal, unicode normalisation |
| `test_ingestion.py` | Ligatures, de-hyphenation, page furniture, headings, blank pages, running headers, corrupt files, OCR probing |
| `test_chunking.py` | Token sizing, no-page-spanning invariant, overlap, table indivisibility, metadata round-trip, all three experiment configs |
| `test_retrieval.py` | Upsert/delete/has-document, dedup, threshold filtering, page validity of citations, citation grounding |
| `test_evaluation.py` | Every metric including edge cases (no gold facts, no citation expectation, malformed JSONL), dataset coverage invariants |
| `test_pipeline.py` | End-to-end ingestion → retrieval → answer, idempotency, prompt-injection containment |

---

## Deployment (Streamlit Community Cloud)

1. **Push the repository** to GitHub. `.gitignore` excludes `.env`, the vector
   database, uploaded PDFs and benchmark outputs; only the synthetic demo corpus
   and the evaluation dataset are committed.

2. **Create the app** and point it at the repo, `app.py`, with
   `requirements.txt` as the dependency file. Do **not** add
   `requirements-optional.txt` unless you want the ~2.5 GB torch download.

3. **Add secrets** under *Settings → Secrets*:

   ```
   LLM_API_KEY = "sk-..."
   LLM_BASE_URL = "https://openrouter.ai/api/v1"
   LLM_MODEL = "openai/gpt-4o-mini"
   ```

   `src/config.py` reads environment variables, and Streamlit surfaces secrets
   as environment variables, so no code change is needed.

4. **Model download size.** The default ONNX MiniLM is ~79 MB and is cached under
   the user's home directory after first use. First request is slow.

### Persistence — read this before deploying

Community Cloud instances run on **ephemeral filesystems**. The ChromaDB index is
written to `./data/chroma_db` and **will be lost on every restart, redeploy, or
idle shutdown.** Users must re-upload their documents afterwards.

If you need durable storage, point `CHROMA_PERSIST_DIR` at a mounted volume
(Streamlit Community Cloud paid tiers support persistent disk), or move to a
host with real storage. The app does not pretend otherwise: the document list
simply comes back empty after a restart.

### Storage limits

The upload limit is enforced in application code (`MAX_UPLOAD_MB`, default 50 MB).
Community Cloud also enforces an overall disk quota — a few hundred MB of PDFs
plus their embeddings will exhaust it. Large corpora belong on a real host.

---

## Troubleshooting

**`File does not look like a PDF (missing %PDF- header)`**
The upload is not a PDF, or it is corrupt. Validation sniffs content, so renaming
a file does not bypass this.

**Everything answers "OCR unavailable"**
Neither Tesseract nor RapidOCR is usable. Check *Settings → System status* for
the exact reason. Install Tesseract and set `TESSERACT_CMD`, or check that
`rapidocr-onnxruntime` imports cleanly (`python -c "import rapidocr_onnxruntime"`).

**Everything answers "not configured - retrieval-only mode"**
`LLM_API_KEY` is unset. Copy `.env.example` to `.env` and fill it in, then
restart the app.

**The scan PDF fails to ingest**
Expected when no OCR backend is present. It is reported in the ingestion log and
counted in `ingestion_failure_rate` — it is not skipped silently.

**The app is slow on first run**
The embedding model (~79 MB) downloads once, then is cached.

**Chroma logs `Failed to send telemetry event ... capture() takes 1 positional
argument`**
A known posthog/chromadb version mismatch. Telemetry is disabled and the logger
is silenced; it does not affect correctness.

---

## Limitations

- **Evaluation corpus is synthetic.** The demo PDFs are generated by
  `scripts/make_demo_corpus.py`, not real government documents. Absolute scores
  do not transfer to real RBI circulars or PMGSY guidelines. The *harness* is
  real and reusable — point it at real documents and re-measure.
- **Retrieval is bi-encoder.** Chunk embeddings are computed independently of
  the query. A cross-encoder reranker is implemented and benchmarkable but off by
  default, because it cannot be shown to help on this corpus.
- **Chunk-size insensitivity is an artefact of small documents** (see
  [Measured results](#measured-results)).
- **Abstention is weak in extractive mode** (0.25). The LLM path is expected to
  abstain more reliably, but this was **not** measured — no API key was available
  in this environment, so `--mode llm` numbers are not reported.
- **OCR was not exercised.** No OCR backend could run on the development machine
  (see below), so the `scanned` questions fail by construction. The code paths
  are implemented and unit-tested, but end-to-end OCR quality is unverified.
- **Table extraction relies on PyMuPDF's `find_tables`**, which handles clean
  bordered tables and gives up on dense or borderless layouts. A failed table is
  logged and skipped rather than aborting the document.
- **No multi-user isolation.** All users share one index. That is fine for a
  single-tenant deployment; it would need per-session collections otherwise.
- **Security posture**: uploads are validated by content, sanitised to a safe
  basename, and never written to disk before validation. Document text is treated
  as untrusted data in the prompt. There is no authentication, so the app should
  not be exposed publicly without one.

### Development-machine note

This project was built and measured on a Windows workstation whose Smart App
Control / WDAC policy blocks unsigned native extensions (`torch`, `pandas`,
`pytesseract`, `rapidocr`'s `shapely`) from loading. That shaped two decisions
that are nevertheless good ones on their own merits:

- The **ONNX embedding path is the default** rather than sentence-transformers —
  it needs no torch and runs anywhere.
- **OCR could not be executed**, so its results are absent from the benchmark.

Neither is expected on a normal machine or on Community Cloud. The affected
paths are implemented, and `requirements-optional.txt` restores the heavier
dependencies where they are permitted.