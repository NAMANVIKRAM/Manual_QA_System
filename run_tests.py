"""
run_tests.py — Standalone test script for the NASA SE Handbook RAG system.
Runs 20 queries, collects retrieved sections/pages/chapters, cross-chapter info,
multihop hops count, and full LLM answer. Saves to test_results.txt.
"""

import os
import re
import sys

sys.path.insert(0, "/Users/namanvikram/Downloads/Manual QA System")

from dotenv import load_dotenv
load_dotenv("/Users/namanvikram/Downloads/Manual QA System/.env")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from acronyms import ACRONYM_MAP
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
CHROMA_DIR = "/Users/namanvikram/Downloads/Manual QA System/chroma_db"
COLLECTION_NAME = "nasa_se_handbook"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
OUTPUT_FILE = "/Users/namanvikram/Downloads/Manual QA System/test_results.txt"

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

QUERIES = [
    # TIER 1 - Baseline
    ("Q1",  "TIER 1 - Baseline",   "What is the purpose of the NASA Systems Engineering Handbook as stated in Section 1.1?"),
    ("Q2",  "TIER 1 - Baseline",   "What are all the entry criteria and exit criteria for Phase B: Preliminary Design and Technology Completion?"),
    ("Q3",  "TIER 1 - Baseline",   "What is the definition of Product Validation and how does it differ from Product Verification?"),
    ("Q4",  "TIER 1 - Baseline",   "List all 17 SE processes defined in the Systems Engineering Engine."),
    ("Q5",  "TIER 1 - Baseline",   "What is the purpose and expected result of a System Requirements Review (SRR) according to Table 6.7.1?"),
    # TIER 2 - Cross-section
    ("Q6",  "TIER 2 - Cross-section", "How does Technical Risk Management (Section 6.4) feed into the Decision Analysis process (Section 6.8)?"),
    ("Q7",  "TIER 2 - Cross-section", "Which SE processes from Chapter 4 are active during Phase C: Final Design and Fabrication?"),
    ("Q8",  "TIER 2 - Cross-section", "How do stakeholder expectations flow from Section 4.1 into technical requirements in Section 4.2 and then into the design solution in Section 4.4?"),
    ("Q9",  "TIER 2 - Cross-section", "Which lifecycle reviews are conducted between Phase B and Phase C, and what are their success criteria?"),
    ("Q10", "TIER 2 - Cross-section", "How does Configuration Management (Section 6.5) interact with Interface Management (Section 6.3)?"),
    # TIER 3 - Acronyms
    ("Q11", "TIER 3 - Acronyms",   "What is the role of HSI in the SE process?"),
    ("Q12", "TIER 3 - Acronyms",   "What happens at a KDP and who makes the decision?"),
    ("Q13", "TIER 3 - Acronyms",   "What is the difference between a PDR and an SDR?"),
    # TIER 4 - Diagrams
    ("Q14", "TIER 4 - Diagrams",   "What does the Systems Engineering Engine diagram show, and what are the three rings or layers?"),
    ("Q15", "TIER 4 - Diagrams",   "Describe the NASA Project Life Cycle process flow shown in Figure 2.2.1."),
    # TIER 5 - Precision
    ("Q16", "TIER 5 - Precision",  "What does the handbook say about the relationship between cost and requirements stability? Give the exact section and paragraph."),
    ("Q17", "TIER 5 - Precision",  "What is NASA's budget for the Artemis program?"),
    ("Q18", "TIER 5 - Precision",  "Does the handbook recommend a specific number of risk categories for the risk matrix?"),
    # TIER 6 - Multi-hop
    ("Q19", "TIER 6 - Multi-hop",  "A project is in Phase D. A critical interface has failed verification. Using the handbook, trace the full decision path: what verification process applies, how the failure should be managed as a risk, and what technical review would be triggered?"),
    ("Q20", "TIER 6 - Multi-hop",  "Section 6.6 covers Technical Data Management. What parent process does it belong to, and how does the data it manages get used as inputs in Section 6.7 Technical Assessment?"),
]

# ---------------------------------------------------------------------------
# Load vector store
# ---------------------------------------------------------------------------

print("Loading embeddings model (this may take a moment)...")
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("Loading vector store...")
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_fn,
    collection_name=COLLECTION_NAME,
)
print("Vector store loaded.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Acronym expansion
# ---------------------------------------------------------------------------

def expand_acronyms(query: str) -> str:
    expanded = query
    for acronym, full_form in ACRONYM_MAP.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        expanded = re.sub(pattern, f"{full_form} ({acronym})", expanded)
    return expanded

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = TOP_K) -> list:
    results = db.similarity_search_with_relevance_scores(query, k=k)
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(score, 3)
        docs.append(doc)
    return docs

def retrieve_multihop(query: str, initial_docs: list) -> tuple[list, int]:
    """Returns (combined_docs, num_extra_hops)."""
    ref_pattern = re.compile(r"[Ss]ection\s+(\d+\.\d+(?:\.\d+){0,2})")
    seen_ids = {d.metadata.get("section_id") for d in initial_docs}
    follow_up_sections = set()

    for doc in initial_docs:
        for m in ref_pattern.finditer(doc.page_content):
            sid = m.group(1)
            if sid not in seen_ids:
                follow_up_sections.add(sid)

    if not follow_up_sections:
        return initial_docs, 0

    extra_docs = []
    for sid in list(follow_up_sections)[:2]:
        hop_query = f"section {sid} {query}"
        results = db.similarity_search_with_relevance_scores(hop_query, k=2)
        for doc, score in results:
            if doc.metadata.get("section_id") not in seen_ids:
                doc.metadata["score"] = round(score, 3)
                doc.metadata["hop"] = True
                extra_docs.append(doc)
                seen_ids.add(doc.metadata.get("section_id"))

    return initial_docs + extra_docs, len(extra_docs)

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        section = meta.get("section_id", "N/A")
        page = meta.get("page", "N/A")
        score = meta.get("score", "")
        hop = " [cross-reference]" if meta.get("hop") else ""
        score_str = f", relevance={score}" if score != "" else ""
        parts.append(
            f"--- Chunk {i} [Section {section}, p.{page}{score_str}]{hop} ---\n{doc.page_content}"
        )
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# System prompt (mirrors app.py)
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

NASA Acronym Reference:
{ACRONYM_GLOSSARY}
"""

# ---------------------------------------------------------------------------
# Groq LLM call
# ---------------------------------------------------------------------------

def call_groq(context: str, question: str) -> str:
    if not GROQ_API_KEY:
        return "ERROR: GROQ_API_KEY not set."
    try:
        user_text = f"Context (extracted text):\n{context}\n\nQuestion: {question}"
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: Groq API call failed: {e}"

# ---------------------------------------------------------------------------
# Run all queries
# ---------------------------------------------------------------------------

def get_chapter(section_id: str) -> str:
    """Return the top-level chapter number from a section_id like '4.2.1' -> '4'."""
    if section_id and section_id != "N/A":
        return section_id.split(".")[0]
    return "N/A"

results_lines = []

for qnum, tier, query in QUERIES:
    print(f"\nRunning {qnum} [{tier}] ...")

    # 1. Expand acronyms
    expanded_query = expand_acronyms(query)

    # 2. Retrieve top-5
    docs = retrieve(expanded_query)

    # 3. Multi-hop
    docs, num_hops = retrieve_multihop(expanded_query, docs)

    # 4. Collect metadata
    sections = []
    pages = []
    chapters = []
    for doc in docs:
        sec = doc.metadata.get("section_id", "N/A")
        pg = doc.metadata.get("page", "N/A")
        ch = get_chapter(sec)
        if sec not in sections:
            sections.append(sec)
        if pg not in pages:
            pages.append(pg)
        if ch not in chapters:
            chapters.append(ch)

    cross_chapter = "YES" if len(chapters) > 1 else "NO"

    # 5. Build context
    context = build_context(docs)

    # 6. Call LLM
    answer = call_groq(context, expanded_query)

    print(f"  Sections: {sections}")
    print(f"  Pages: {pages}")
    print(f"  Chapters: {chapters}  Cross-chapter: {cross_chapter}")
    print(f"  Hops: {num_hops}")
    print(f"  Answer (first 120 chars): {answer[:120]}")

    # 7. Format output block
    block = []
    block.append("=" * 80)
    block.append(f"{qnum} [{tier}] - {query}")
    block.append("-" * 80)
    block.append(f"RETRIEVED SECTIONS: {sections}")
    block.append(f"PAGES: {pages}")
    block.append(f"CHAPTERS: {chapters}  CROSS-CHAPTER: {cross_chapter}")
    block.append(f"HOPS: {num_hops} extra docs")
    block.append("-" * 80)
    block.append("ANSWER:")
    block.append(answer)
    block.append("=" * 80)
    block.append("")

    results_lines.extend(block)

# ---------------------------------------------------------------------------
# Write to file
# ---------------------------------------------------------------------------

output_text = "\n".join(results_lines)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"\nDone. Results written to {OUTPUT_FILE}")
