"""
app.py — Gradio chatbot UI + query pipeline for the NASA SE Handbook QA system.

Query pipeline per question:
  1. Expand acronyms in the user question
  2. Retrieve top-k=5 semantically similar chunks from ChromaDB (with scores)
  3. Multi-hop: follow cross-references found in retrieved chunks
  4. Detect if query needs vision (diagram/figure/table question)
  5. If vision needed: render relevant PDF pages as images → pass to Llama 4 Scout
  6. Build prompt with mandatory citation instructions → call LLM
  7. Return answer + sources panel (section, page, score, type)

LLM: Groq — meta-llama/llama-4-scout-17b-16e-instruct (vision + text)

Run:
    python app.py
Then open http://localhost:7860
"""

import base64
import io
import os
import re
import threading

import gradio as gr
import requests
from dotenv import load_dotenv

from acronyms import ACRONYM_MAP
from evaluate import evaluate_response, format_scores_html

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nasa_se_handbook")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TOP_K = int(os.getenv("TOP_K", "7"))
PDF_PATH = os.getenv("PDF_PATH", "nasa_systems_engineering_handbook_0.pdf")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# HuggingFace Inference API
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

# Vision trigger keywords — queries matching these get page images sent to the LLM
VISION_KEYWORDS = re.compile(
    r"\b(diagram|figure|fig|chart|flowchart|vee model|vee-model|lifecycle|"
    r"life cycle|process flow|table|illustration|visual|image|drawing|schematic|"
    r"engine|layer|ring|figure|fig\.|matrix|list all|enumerate|how many)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Load vector store (once at startup)
# ---------------------------------------------------------------------------


def _load_vectorstore():
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}'. "
            "Run 'python ingest.py' first."
        )
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_fn,
        collection_name=COLLECTION_NAME,
    )


print("Loading vector store...")
try:
    vectorstore = _load_vectorstore()
    print("Vector store loaded.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    vectorstore = None

# ---------------------------------------------------------------------------
# Vision helpers — render PDF pages to base64 images for Llama 4 Scout
# ---------------------------------------------------------------------------


def _page_to_base64(page_number: int, dpi: int = 150) -> str | None:
    """
    Render a single PDF page (1-indexed) to a base64-encoded JPEG.
    Returns None if pdf2image / poppler is not available.
    """
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            PDF_PATH,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            return None
        buf = io.BytesIO()
        images[0].save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def get_page_images_for_docs(docs: list, max_pages: int = 3) -> list[tuple[int, str]]:
    """
    Return up to max_pages unique (page_number, base64_jpeg) pairs
    from the retrieved docs. Deduplicates by page number.
    """
    seen = set()
    results = []
    for doc in docs:
        page = doc.metadata.get("page")
        if page and page not in seen:
            seen.add(page)
            b64 = _page_to_base64(page)
            if b64:
                results.append((page, b64))
        if len(results) >= max_pages:
            break
    return results


# ---------------------------------------------------------------------------
# Acronym expansion
# ---------------------------------------------------------------------------


def expand_acronyms(query: str, acronym_map: dict) -> str:
    """Replace known acronyms with 'Full Form (ACRONYM)' using word-boundary matching."""
    expanded = query
    for acronym, full_form in acronym_map.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        expanded = re.sub(pattern, f"{full_form} ({acronym})", expanded)
    return expanded


# ---------------------------------------------------------------------------
# Retrieval  (returns docs with similarity scores attached as metadata)
# ---------------------------------------------------------------------------


def retrieve(query: str, k: int = TOP_K) -> list:
    """
    Semantic retrieval with similarity scores.
    Uses similarity_search_with_relevance_scores so each doc carries
    a 'score' metadata field (0-1, higher = more relevant).
    """
    if vectorstore is None:
        return []
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(score, 3)
        docs.append(doc)
    return docs


def retrieve_multihop(query: str, initial_docs: list) -> list:
    """
    Multi-hop retrieval:
    Pass 1 — follow explicit "Section X.X.X" cross-references in retrieved text.
    Pass 2 — detect section gaps in multi-section queries (e.g. query mentions
             4.1 and 4.4 but retrieved chunks skip 4.2) and fill them in.
    Pass 3 — Phase-aware expansion: if query mentions a lifecycle phase (A/B/C/D/E/F),
             also fetch the corresponding life-cycle review table chunk.
    Returns deduplicated combined doc list.
    """
    if vectorstore is None:
        return initial_docs

    # Extract "see section X.X.X" style references from retrieved text
    ref_pattern = re.compile(r"[Ss]ection\s+(\d+\.\d+(?:\.\d+){0,2})")
    seen_content = {d.page_content[:100] for d in initial_docs}
    seen_ids = {d.metadata.get("section_id") for d in initial_docs}
    follow_up_sections = set()

    for doc in initial_docs:
        for m in ref_pattern.finditer(doc.page_content):
            sid = m.group(1)
            if sid not in seen_ids:
                follow_up_sections.add(sid)

    # Gap detection: if query explicitly mentions section numbers, check they're covered
    query_sections = set(re.findall(r"\b(\d+\.\d+(?:\.\d+)?)\b", query))
    for sid in query_sections:
        if sid not in seen_ids:
            follow_up_sections.add(sid)

    extra_docs = []

    # Phase-aware expansion: if query asks about lifecycle phase reviews, fetch Table 6.7-1
    phase_review_pattern = re.compile(
        r"\b(?:phase\s+[A-F]|lifecycle\s+review|life.cycle\s+review|KDP|key\s+decision\s+point"
        r"|technical\s+review|design\s+review|PDR|CDR|SRR|SDR|MCR|ORR|SIR|FRR|PRR)\b",
        re.IGNORECASE,
    )
    if phase_review_pattern.search(query):
        # General lifecycle review table fetch
        hop_query = f"life cycle review phase timing purpose results spaceflight {query}"
        phase_results = vectorstore.similarity_search_with_relevance_scores(hop_query, k=4)
        for doc, score in phase_results:
            snippet = doc.page_content[:100]
            if snippet not in seen_content and ("TABLE 6.7-1" in doc.page_content or "life-cycle review" in doc.page_content.lower() or "lifecycle review" in doc.page_content.lower()):
                doc.metadata["score"] = round(score, 3)
                doc.metadata["hop"] = True
                extra_docs.append(doc)
                seen_content.add(snippet)
                seen_ids.add(doc.metadata.get("section_id"))

        # For Phase B queries, also fetch PDR row directly (often missed by semantic search)
        if re.search(r"\bphase\s*B\b", query, re.IGNORECASE):
            pdr_results = vectorstore.similarity_search_with_relevance_scores(
                "Preliminary Design Review PDR purpose timing Phase B results", k=3
            )
            for doc, score in pdr_results:
                snippet = doc.page_content[:100]
                if snippet not in seen_content and "TABLE 6.7-1" in doc.page_content:
                    doc.metadata["score"] = round(score, 3)
                    doc.metadata["hop"] = True
                    extra_docs.append(doc)
                    seen_content.add(snippet)
                    seen_ids.add(doc.metadata.get("section_id"))

        # For Phase C queries, also fetch CDR row directly
        if re.search(r"\bphase\s*C\b", query, re.IGNORECASE):
            cdr_results = vectorstore.similarity_search_with_relevance_scores(
                "Critical Design Review CDR purpose timing Phase C results", k=3
            )
            for doc, score in cdr_results:
                snippet = doc.page_content[:100]
                if snippet not in seen_content and "TABLE 6.7-1" in doc.page_content:
                    doc.metadata["score"] = round(score, 3)
                    doc.metadata["hop"] = True
                    extra_docs.append(doc)
                    seen_content.add(snippet)
                    seen_ids.add(doc.metadata.get("section_id"))

    # SE Engine / Section 2.1 expansion: if query asks about SE Engine layers/rings/sets or 17 processes,
    # fetch the Section 2.1 chunks that contain the numbered process list
    if re.search(r"\b(se\s+engine|section\s+2\.1|three\s+(layer|ring|set)|common\s+technical\s+process|17\s+(se|systems?\s+engineering)\s+process|list\s+(all\s+)?17)\b", query, re.IGNORECASE):
        se_engine_results = vectorstore.similarity_search_with_relevance_scores(
            "Stakeholder Expectations Technical Requirements Logical Decomposition Design Solution Product Implementation Integration Verification Validation Transition Technical Planning Requirements Management Interface Technical Risk Configuration Data Assessment Decision Analysis 17 processes", k=5
        )
        for doc, score in se_engine_results:
            snippet = doc.page_content[:100]
            if snippet not in seen_content and doc.metadata.get("section_id") in ("2.1", "2.2") and len(doc.page_content) > 200:
                doc.metadata["score"] = round(score, 3)
                doc.metadata["hop"] = True
                extra_docs.append(doc)
                seen_content.add(snippet)
                seen_ids.add(doc.metadata.get("section_id"))

    if not follow_up_sections and not extra_docs:
        return initial_docs

    # Retrieve for each referenced section by constructing a targeted query
    for sid in list(follow_up_sections)[:4]:  # cap at 4 extra hops for multi-section queries
        hop_query = f"section {sid} {query}"
        results = vectorstore.similarity_search_with_relevance_scores(hop_query, k=2)
        for doc, score in results:
            snippet = doc.page_content[:100]
            if doc.metadata.get("section_id") not in seen_ids and snippet not in seen_content:
                doc.metadata["score"] = round(score, 3)
                doc.metadata["hop"] = True
                extra_docs.append(doc)
                seen_content.add(snippet)
                seen_ids.add(doc.metadata.get("section_id"))

    return initial_docs + extra_docs


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

ACRONYM_GLOSSARY = "\n".join(f"  {k}: {v}" for k, v in ACRONYM_MAP.items())

SYSTEM_PROMPT = f"""You are a precise technical assistant for the NASA Systems Engineering Handbook (SP-2016-6105 Rev2).

Rules:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge.
2. Every factual claim MUST be followed by a citation in the format [Section X.X.X, p.Y].
   If a claim spans multiple sections, cite all relevant sections.
3. If the retrieved context is insufficient to answer the question, say:
   "The retrieved context does not contain enough information to answer this question."
4. Do not fabricate section numbers or page numbers.
5. Structure your answer clearly. Use bullet points or numbered lists where appropriate.
6. When a question asks about "layers", "rings", or "sets" of a framework, look for the sets or groups
   described in the context — they are the same concept expressed differently. For example, the "three
   layers of the SE Engine" refers to the three sets of common technical processes: system design,
   product realization, and technical management.
7. When the context mentions a figure or table but does not reproduce its full content, describe what
   the context says the figure/table contains rather than claiming insufficient information.

NASA Acronym Reference:
{ACRONYM_GLOSSARY}
"""


def build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        section = meta.get("section_id", "N/A")
        page = meta.get("page", "N/A")
        score = meta.get("score", "")
        hop = " [cross-reference]" if meta.get("hop") else ""
        score_str = f", relevance={score}" if score != "" else ""
        parts.append(f"--- Chunk {i} [Section {section}, p.{page}{score_str}]{hop} ---\n{doc.page_content}")
    return "\n\n".join(parts)


def build_prompt(context: str, question: str) -> str:
    """Build a single-string prompt for models that don't support chat format."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def _call_ollama(prompt: str) -> str:
    """Call Ollama's /api/generate endpoint (streaming=False)."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Could not connect to Ollama at "
            f"{OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {LLM_MODEL}`)."
        )
    except requests.exceptions.HTTPError as e:
        return f"ERROR: Ollama returned an error: {e}"


def _call_huggingface(prompt: str) -> str:
    """Call HuggingFace Inference API (free tier)."""
    if not HF_API_TOKEN:
        return (
            "ERROR: HF_API_TOKEN not set. "
            "Add it to .env to use the HuggingFace Inference API."
        )
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        return str(result)
    except requests.exceptions.HTTPError as e:
        return f"ERROR: HuggingFace API returned an error: {e}\n{resp.text}"


def _call_groq(context: str, question: str, page_images: list | None = None) -> str:
    """
    Call Groq with Llama 4 Scout.
    If page_images is provided (list of base64 JPEGs), sends a multimodal
    message so the model can see diagrams, figures, and tables directly.
    """
    if not GROQ_API_KEY:
        return "ERROR: GROQ_API_KEY not set. Add it to .env"
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        # Build user message content
        user_text = f"Context (extracted text):\n{context}\n\nQuestion: {question}"

        if page_images:
            # Multimodal: interleave text + images
            content = [{"type": "text", "text": user_text}]
            for page_num, b64 in page_images:
                content.append({
                    "type": "text",
                    "text": f"\n[Page {page_num} image from the handbook:]"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })
        else:
            content = user_text

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: Groq API call failed: {e}"


def ask_llm(context: str, question: str, page_images: list | None = None) -> str:
    if LLM_PROVIDER == "groq":
        return _call_groq(context, question, page_images)
    elif LLM_PROVIDER == "huggingface":
        prompt = build_prompt(context, question)
        return _call_huggingface(prompt)
    else:
        prompt = build_prompt(context, question)
        return _call_ollama(prompt)


# ---------------------------------------------------------------------------
# Sources panel
# ---------------------------------------------------------------------------


def format_sources(docs: list) -> str:
    if not docs:
        return "<p><em>No sources retrieved.</em></p>"

    rows = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        section = meta.get("section_id", "N/A")
        parent = ".".join(section.split(".")[:2]) if section and section != "N/A" else "N/A"
        page = meta.get("page", "N/A")
        score = meta.get("score", "")
        hop = meta.get("hop", False)
        score_str = f"{score:.0%}" if isinstance(score, float) else (str(score) if score != "" else "-")
        label = "🔗 cross-ref" if hop else "primary"
        snippet = doc.page_content[:180].replace("\n", " ").strip()
        if len(doc.page_content) > 180:
            snippet += "…"
        row_bg = "#fffbe6" if hop else "white"
        score_color = "#c07000" if hop else "#2a7a2a"
        rows.append(
            f"<tr style='background:{row_bg};'>"
            f"<td style='padding:4px 8px;color:#666;'>{i}</td>"
            f"<td style='padding:4px 8px;font-weight:bold;'>§{section}</td>"
            f"<td style='padding:4px 8px;color:#888;font-size:0.8em;'>↑§{parent}</td>"
            f"<td style='padding:4px 8px;'>p.{page}</td>"
            f"<td style='padding:4px 8px;font-weight:bold;color:{score_color};'>{score_str}</td>"
            f"<td style='padding:4px 8px;font-size:0.75em;color:#888;'>{label}</td>"
            f"<td style='padding:4px 8px;font-size:0.85em;color:#444;'>{snippet}</td>"
            f"</tr>"
        )

    table = (
        "<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:0.9em;'>"
        "<thead><tr style='background:#f0f0f0;'>"
        "<th style='padding:4px 8px;text-align:left;'>#</th>"
        "<th style='padding:4px 8px;text-align:left;'>Section</th>"
        "<th style='padding:4px 8px;text-align:left;'>Parent</th>"
        "<th style='padding:4px 8px;text-align:left;'>Page</th>"
        "<th style='padding:4px 8px;text-align:left;'>Score</th>"
        "<th style='padding:4px 8px;text-align:left;'>Type</th>"
        "<th style='padding:4px 8px;text-align:left;'>Snippet</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return f"<details open><summary><strong>Sources ({len(docs)} retrieved)</strong></summary>{table}</details>"


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------


def chat(user_message: str, history: list) -> tuple:
    if not user_message.strip():
        return "", history, ""

    expanded_query = expand_acronyms(user_message, ACRONYM_MAP)
    docs = retrieve(expanded_query)
    docs = retrieve_multihop(expanded_query, docs)
    context = build_context(docs)

    # Vision: if query mentions diagrams/figures/tables, send page images too
    page_images = None
    if VISION_KEYWORDS.search(user_message):
        page_images = get_page_images_for_docs(docs, max_pages=3)

    answer = ask_llm(context, expanded_query, page_images)
    sources_html = format_sources(docs)

    # Ragas evaluation — run in background thread so UI isn't blocked
    # Result stored in a list so the thread can write to it
    ragas_result = [{}]

    def _run_ragas():
        contexts = [doc.page_content for doc in docs]
        scores = evaluate_response(user_message, answer, contexts)
        ragas_result[0] = scores

    t = threading.Thread(target=_run_ragas, daemon=True)
    t.start()
    t.join(timeout=60)   # wait up to 60s for scores; skip if too slow

    eval_html = format_scores_html(ragas_result[0])
    combined_sources = sources_html + eval_html

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]

    return "", history, combined_sources


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What is the purpose of the NASA Systems Engineering Handbook?",
    "What is a Technology Readiness Level and how is TRL 6 defined?",
    "What are the entry and exit criteria for PDR?",
    "What is the difference between verification and validation?",
    "List all 17 SE processes defined in the Systems Engineering Engine.",
    "What happens at a KDP and who makes the decision?",
    "How does Technical Risk Management feed into the Decision Analysis process?",
    "What is the role of HSI in the SE process?",
    "Which lifecycle reviews are conducted between Phase B and Phase C?",
    "How does Configuration Management interact with Interface Management?",
]

_provider_label = {
    "groq": f"Groq · {LLM_MODEL} (vision + text)",
    "ollama": f"Ollama · {LLM_MODEL}",
    "huggingface": f"HuggingFace · {LLM_MODEL}",
}.get(LLM_PROVIDER, LLM_PROVIDER)

with gr.Blocks(title="NASA SE Handbook — QA System") as demo:

    gr.Markdown(
        f"""
# NASA Systems Engineering Handbook — QA System
Ask questions about the **NASA SE Handbook (SP-2016-6105 Rev2)**.
Every answer includes section-level citations.

**Embeddings:** `{EMBEDDING_MODEL}` (HuggingFace) &nbsp;|&nbsp; **LLM:** `{_provider_label}`
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=480)
            sources_panel = gr.HTML(
                value="<p style='color:#888;'>Sources will appear here after your first question.</p>",
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask a question about the NASA SE Handbook...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(examples=EXAMPLE_QUERIES, inputs=user_input, label="Example queries")

    submit_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, sources_panel],
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, sources_panel],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
