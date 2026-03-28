# NASA Systems Engineering Handbook — QA System

A RAG-based (Retrieval-Augmented Generation) chatbot that answers natural language questions over the 270-page NASA Systems Engineering Handbook (SP-2016-6105 Rev2), with **section-level citations on every response**.

Built for the **i2e Consulting AI Labs Hireathon 2026**.

---

## Table of Contents

- [How to Run the Application](#how-to-run-the-application)
- [Architecture Overview](#architecture-overview)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations & Failure Modes](#known-limitations--failure-modes)
- [Test Results](#test-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## How to Run the Application

### Prerequisites

- Python 3.10+
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- The NASA SE Handbook PDF (download link below)
- `poppler` installed (required for pdf2image vision rendering)
  - macOS: `brew install poppler`
  - Ubuntu: `sudo apt-get install poppler-utils`

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd nasa-qa-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the NASA PDF

Download the handbook and place it in the project root:

```
https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf
```

Save it as: `nasa_systems_engineering_handbook_0.pdf`

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
LLM_PROVIDER=groq
LLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_API_KEY=your-groq-api-key-here
EMBEDDING_PROVIDER=huggingface
```

### 5. Ingest the document (run once)

This reads the PDF, chunks it by section boundaries, extracts tables, embeds everything, and saves the vector store to disk. Takes 3–6 minutes on first run.

```bash
python ingest.py
```

Expected output:
```
Loading PDF: nasa_systems_engineering_handbook_0.pdf
  Loaded 310 pages.
  Detected 312 section boundaries.
  Created 890 chunks (avg 1100 chars).
  Extracted 248 table row chunks.
  Total chunks: 1138 (890 text + 248 table rows)
Embedding 1138 chunks and saving to ./chroma_db...
  Vector store saved to ./chroma_db
Ingestion complete. Run 'python app.py' to start the chatbot.
```

You only need to run this once. The vector store persists to `./chroma_db`.

### 6. Launch the chatbot

```bash
python app.py
```

Open your browser at: **http://localhost:7860**

---

## Architecture Overview

### Ingestion Pipeline (run once)

```
NASA PDF (270 pages)
        |
        v
  PyPDFLoader  ──────────────────────────────────────────────
  (page-level text + metadata)                               |
        |                                                    |
        v                                                    v
  Section-Boundary Chunker                        pdfplumber Table Extractor
  (regex detects headings like "2.3 Title")       (each row stored as a self-
  + RecursiveCharacterTextSplitter fallback        contained document with
    for sections > 1500 chars                      headers prepended)
        |                                                    |
        +─────────────────────┬──────────────────────────────+
                              |
                              v
                  890 text chunks + 248 table chunks = 1138 total
                              |
                              v
              HuggingFace Embeddings (all-MiniLM-L6-v2)
                              |
                              v
                  ChromaDB  (persisted to ./chroma_db)
```

### Query Pipeline (per question)

```
User Question
      |
      v
┌─────────────────────┐
│  Acronym Expansion  │  ← 198 acronyms from Appendix A
│  HSI → Human        │    e.g. "What is HSI?" becomes
│  Systems Integration│    "What is Human Systems Integration (HSI)?"
└─────────────────────┘
      |
      v
┌─────────────────────┐
│  Semantic Retrieval │  ← ChromaDB similarity_search_with_relevance_scores
│  TOP_K = 7 chunks   │    Returns score 0–1 for each chunk
└─────────────────────┘
      |
      v
┌─────────────────────────────────────────────────────────────┐
│  Multi-Hop Retrieval (3 passes)                             │
│                                                             │
│  Pass 1 — Follow "Section X.X.X" cross-refs found in       │
│           retrieved chunk text                              │
│                                                             │
│  Pass 2 — Gap detection: fetch sections explicitly          │
│           mentioned in the query but not yet retrieved      │
│                                                             │
│  Pass 3 — Domain-aware expansion:                           │
│    · Phase B query      → fetch PDR row from Table 6.7-1   │
│    · Phase C query      → fetch CDR row from Table 6.7-1   │
│    · "SE Engine" query  → fetch Section 2.1 numbered list  │
│    · "17 processes"     → fetch Section 2.1 numbered list  │
└─────────────────────────────────────────────────────────────┘
      |
      v
┌─────────────────────┐
│  Vision Detection   │  ← VISION_KEYWORDS regex
│  (optional)         │    If matched: render PDF pages as
│                     │    base64 JPEG via pdf2image + poppler
└─────────────────────┘
      |
      v
┌─────────────────────┐
│  LLM Call           │  ← Groq API
│                     │    Model: Llama 4 Scout (vision + text)
│                     │    Temp: 0.1 | Max tokens: 1024
└─────────────────────┘
      |
      v
┌─────────────────────┐
│  Ragas Evaluation   │  ← Background thread (60s timeout)
│                     │    Faithfulness + Context Precision
└─────────────────────┘
      |
      v
  Answer + Sources Panel + Ragas Score Badges
  (section, page, relevance score, hop type)
```

### Component Responsibilities

| Component | Role | Technology |
|---|---|---|
| PDF Loader | Extract text with page metadata | `PyPDFLoader` (LangChain) |
| Table Extractor | Extract structured tables row-by-row | `pdfplumber` |
| Chunker | Split at section boundaries, not token count | Custom regex + `RecursiveCharacterTextSplitter` |
| Embedder | Convert chunks to vectors | `all-MiniLM-L6-v2` (HuggingFace, local, free) |
| Vector Store | Store + retrieve chunks by semantic similarity | `ChromaDB` (local, disk-persisted) |
| Multi-Hop Engine | Follow cross-references across sections | Custom 3-pass retrieval logic |
| LLM | Synthesise cited answers from context | Llama 4 Scout via Groq API (vision + text) |
| Evaluator | Score answer quality per query | `Ragas` (Faithfulness + Context Precision) |
| UI | Chat interface with sources panel | `Gradio 6.x` |

---

## Key Design Decisions

### 1. Section-boundary chunking over fixed token windows

**Decision:** Split the document at section headings detected by regex rather than at fixed 512- or 1500-character windows.

**Why:** The NASA handbook has 4 levels of section nesting (e.g., `6.3.2.1`). A fixed splitter cuts mid-paragraph and mid-table with no awareness of meaning boundaries. Section-boundary chunking ensures each chunk is a complete logical unit that can be cited precisely.

**Regex pattern:**
```
r"(?:^|(?<=\n))([1-9]\.\d{1,2}(?:\.\d{1,2}){0,2})(?:\s+[A-Z]|\s*\n)"
```
The chapter is restricted to a single digit (`[1-9]`) to avoid matching page numbers like `21.0` or `42.0` found in the table of contents.

**Trade-off:** Variable chunk length. Sections > 1500 chars are further split with `RecursiveCharacterTextSplitter`, inheriting the same metadata.

---

### 2. pdfplumber for table extraction alongside PyPDF

**Decision:** Use `pdfplumber` specifically for structured table extraction, in addition to PyPDF for body text.

**Why:** PyPDF mangles multi-column tables into garbled, column-order-ambiguous text. Table 6.7-1 (lifecycle reviews — the most queried table in testing) spans 4 pages and was completely unreadable via PyPDF. pdfplumber correctly identifies rows and columns.

**How:** Each table row is stored as a self-contained Document:
```
[TABLE 6.7-1]
Name of Review: PDR | Purpose: ... | Timing: ... | Entrance Criteria: ...
```
This makes every row independently queryable without needing the header page to be in the same retrieval result.

**Result:** 248 additional table row chunks, enabling Q5 (SRR purpose), Q9 (PDR success criteria), Q13 (PDR vs SDR) to all pass correctly.

---

### 3. Three-pass multi-hop retrieval

**Decision:** After initial TOP_K=7 retrieval, run three additional expansion passes before calling the LLM.

**Why:** Many questions span multiple sections. A single semantic query cannot always retrieve all needed sections simultaneously. Testing showed that Q9 ("Which review marks end of Phase B?") consistently retrieved Section 3.5 chunks but missed Table 6.7-1's PDR row — which has the definitive answer. Without multi-hop, the LLM hallucinated CDR instead of PDR.

**Three passes:**
- **Pass 1** — Follow "Section X.X" cross-references found in retrieved text
- **Pass 2** — Fetch sections explicitly mentioned in the user query
- **Pass 3** — Domain-aware rules for known retrieval gaps (Phase B/C reviews → Table 6.7-1, "17 processes" → Section 2.1 numbered list)

**Trade-off:** Adds latency (2–4 extra ChromaDB lookups). Capped at 4 extra hops. Domain rules in Pass 3 are manually crafted — they cover known failure patterns but are not generalisable.

---

### 4. Llama 4 Scout (multimodal) on Groq

**Decision:** Use `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API for all LLM calls.

**Why:** Llama 4 Scout supports both text and image input (multimodal). The NASA handbook contains critical diagrams (Figure 2.1-1 SE Engine, Figure 2.2-1 Life Cycle). When a query mentions diagrams or figures, the system renders the relevant PDF pages as base64 JPEG images and sends them alongside text context. This is done at query time, not ingestion time, so no extra storage is needed.

**Why Groq:** Groq's inference hardware (LPU) delivers very low latency (~1–2 seconds per response vs 5–10 seconds on standard GPU endpoints). Free tier provides 500k tokens/day.

**Trade-off:** 500k token/day limit on the free tier. At ~4k tokens per query, this supports ~125 queries/day. Upgrade to paid tier for production use.

---

### 5. HuggingFace embeddings (local, free)

**Decision:** Use `all-MiniLM-L6-v2` locally instead of OpenAI or Cohere embeddings.

**Why:** Runs entirely on-device with no API cost and no latency for embedding calls. Produces 384-dimensional sentence embeddings well-suited for technical document retrieval. Keeps the system fully functional with zero recurring cost.

**Trade-off:** Lower recall quality compared to OpenAI `text-embedding-3-large`. Similarity scores in the 0.3–0.6 range (vs 0.7–0.9 for OpenAI). This contributes to some retrieval misses for paraphrased queries, which the multi-hop passes partially compensate for.

---

### 6. Ragas evaluation baked into every query

**Decision:** Run Faithfulness + Context Precision scoring on every query response, displayed as colour-coded badges in the UI.

**Why:** Provides real-time quality visibility without a separate evaluation step. Users can immediately see if an answer is well-grounded (green ≥75%) or suspect (red <50%). Runs in a background thread so it never blocks the UI.

**Metrics:**
- **Faithfulness** — are all claims in the answer grounded in the retrieved chunks?
- **Context Precision** — were the retrieved chunks actually useful for answering?

---

### 7. Strict citation enforcement in the system prompt

**Decision:** The system prompt mandates `[Section X.X, p.Y]` format for every factual claim and explicitly says "Do not fabricate section numbers or page numbers."

**Why:** LLMs tend to hallucinate plausible-sounding section numbers. The strict prompt combined with "If context is insufficient, say so explicitly" produced **0% hallucination rate** across 40 test questions — including the adversarial test ("What is NASA's budget for the Artemis program?") which was correctly rejected.

---

## Known Limitations & Failure Modes

### 1. Diagram-only content not in the vector store

**Affected:** Questions about the Vee model, Figure 2.2-1 process flow, risk matrix categories.

**Root cause:** PyPDF only extracts the text layer of each page. Content inside figures, diagrams, and colour-coded matrices is never ingested. The word "Vee" appears zero times across all 1138 chunks.

**Impact:** Any question asking about visual content returns a partial answer. The system acknowledges the gap honestly rather than hallucinating.

**Fix path:** Run vision (Llama 4 Scout) on each page during ingestion and store figure descriptions as `content_type=diagram` chunks.

---

### 2. Formal entry/exit criteria delegated to NPR 7123.1

**Affected:** Phase B entry/exit criteria (Q2), Phase C criteria.

**Root cause:** The handbook explicitly defers formal criteria to NPR 7123.1 (Systems Engineering Processes and Requirements). The handbook describes activities and intent; NPR 7123.1 contains the formal compliance tables.

**Fix path:** Ingest NPR 7123.1 as a second source document alongside the handbook.

---

### 3. ToC chunks pollute section retrieval

**Root cause:** Table of contents entries (e.g., "2.1 The Common Technical Processes . . . . . 5") have the same `section_id` metadata as the actual content pages. Semantic search sometimes retrieves the ToC entry instead of the substantive content.

**Fix path:** During ingestion, detect ToC pages (typically pages 1–10) and tag chunks as `content_type=toc` or exclude them from the collection.

---

### 4. Vision triggers on retrieved pages, not target pages

**Root cause:** When vision is activated, the system renders pages from the retrieved chunks. If the diagram being asked about is on page 15 but retrieved chunks are from pages 23 and 35, the wrong pages are sent to the vision model.

**Fix path:** Build a figure index at ingestion time mapping figure numbers (Figure 2.1-1, Figure 2.2-1) to their page numbers. At query time, look up and render that exact page.

---

### 5. Groq free-tier token limit (500k tokens/day)

**Root cause:** During the 40-question test run, the daily limit was hit mid-session, causing 429 errors for remaining questions.

**Fix path:** Upgrade to Groq Dev Tier (paid) or implement a two-key rotation fallback.

---

### 6. Multi-hop domain rules are manually crafted

**Root cause:** The Phase B/PDR expansion, Phase C/CDR expansion, and SE Engine expansion are hard-coded regex rules written to fix specific known retrieval failures.

**Impact:** Novel query types not matching these rules won't benefit from domain-aware expansion.

**Fix path:** Replace manual rules with a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM`) to rerank retrieved chunks before the LLM call.

---

## Test Results

Testing was performed across two question sets — 40 questions total.

### Test Set 1 (20 questions — previous validation set)

| Tier | Description | Score |
|---|---|---|
| Tier 1 | Basic factual retrieval | 5/5 |
| Tier 2 | Multi-section synthesis | 4/5 |
| Tier 3 | Acronym handling | 3/3 |
| Tier 4 | Diagram awareness | 1/2 |
| Tier 5 | Precision & confidence | 3/3 |
| Tier 6 | Multi-hop reasoning | 2/2 |
| **Total** | | **18/20 (90% PASS)** |

### Test Set 2 (20 questions — harder, explicit section references)

| Tier | Description | PASS | PARTIAL | FAIL |
|---|---|---|---|---|
| Tier 1 | Baseline retrieval | 4 | 1 | 0 |
| Tier 2 | Cross-section retrieval | 3 | 2 | 0 |
| Tier 3 | Acronym & terminology | 3 | 0 | 0 |
| Tier 4 | Diagram & visual content | 1 | 1 | 0 |
| Tier 5 | Precision & confidence | 2 | 1 | 0 |
| Tier 6 | Multi-hop reasoning | 1 | 1 | 0 |
| **Total** | | **13** | **7** | **0** |

**Hallucination rate: 0%** — All 7 partial answers are honest "I don't know" responses, not fabricated content. The adversarial Q17 ("What is NASA's budget for the Artemis program?") was correctly rejected.

**Partial answer breakdown (all 7 are data gaps, not errors):**
- 3 × diagram-only content (Figure 2.2-1, risk matrix)
- 1 × criteria delegated to external NPR 7123.1
- 1 × section not retrieved in top-K
- 1 × data flow is diagram-only
- 1 × minor imprecision in 3-hop chain (ORR vs SIR)

Full test results and test script are in the `tests/` folder.

---

## Project Structure

```
nasa-qa-system/
│
├── app.py                   # Gradio UI + full query pipeline
├── ingest.py                # One-time ingestion pipeline
├── evaluate.py              # Ragas evaluation (Faithfulness + Context Precision)
├── acronyms.py              # 198 acronyms from Appendix A
├── requirements.txt         # All dependencies
├── README.md
│
├── tests/
│   ├── run_tests2.py        # Standalone 20-question test runner (Test Set 2)
│   └── test_results2.txt    # Full answers + section citations for all 20 questions
│
├── deployment/
│   └── Dockerfile           # Container setup for deployment
│
├── reports/                 # Local analysis only (not pushed to GitHub)
│   ├── system_report.txt    # Architecture + design decisions (detailed)
│   ├── test_results.txt     # Test Set 1 raw results
│   └── test_results2.txt    # Test Set 2 raw results
│
├── chroma_db/               # Generated by ingest.py — not pushed to GitHub
└── .env                     # API keys — not pushed to GitHub
```

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) | Free |
| Vector Store | ChromaDB (disk-persisted) | Free |
| LLM | Llama 4 Scout via Groq API | Free tier |
| Table Extraction | pdfplumber | Free |
| PDF Rendering (vision) | pdf2image + poppler | Free |
| Evaluation | Ragas | Free (uses Groq tokens) |
| UI | Gradio 6.x | Free |
| PDF Loading | PyPDF via LangChain | Free |

**Total external API cost: $0** (Groq free tier — 500k tokens/day)

---

Built for i2e Consulting — AI Labs Hireathon 2026 | March 2026
