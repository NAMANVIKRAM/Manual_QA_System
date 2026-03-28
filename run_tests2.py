"""
run_tests2.py — Standalone test runner for NASA SE Handbook RAG system.
Tests 20 queries across 6 tiers. Does NOT import from app.py.
"""

import os
import re
import sys

os.chdir("/Users/namanvikram/Downloads/Manual QA System")
sys.path.insert(0, "/Users/namanvikram/Downloads/Manual QA System")

from dotenv import load_dotenv
load_dotenv("/Users/namanvikram/Downloads/Manual QA System/.env")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from acronyms import ACRONYM_MAP
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TOP_K = 5

print("Loading vector store...")
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    collection_name="nasa_se_handbook",
)
print("Vector store loaded.")

client = Groq(api_key=GROQ_API_KEY)

ACRONYM_GLOSSARY = "\n".join(f"  {k}: {v}" for k, v in ACRONYM_MAP.items())

SYSTEM_PROMPT = f"""You are a precise technical assistant for the NASA Systems Engineering Handbook (SP-2016-6105 Rev2).
Rules:
1. Answer ONLY using the provided context chunks.
2. Every factual claim MUST be followed by a citation [Section X.X.X, p.Y].
3. If context is insufficient, say so explicitly.
4. Do not fabricate section numbers or page numbers.
5. Structure answers clearly with bullet points where appropriate.

NASA Acronym Reference:
{ACRONYM_GLOSSARY}
"""


def expand_acronyms(query: str) -> str:
    expanded = query
    for acronym, full_form in ACRONYM_MAP.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        expanded = re.sub(pattern, f"{full_form} ({acronym})", expanded)
    return expanded


def retrieve_and_multihop(query: str, original_query: str) -> tuple:
    """
    Returns (docs, hop_count).
    docs: list of (doc, score) tuples.
    """
    # Initial retrieval
    results = db.similarity_search_with_relevance_scores(query, k=TOP_K)
    docs_with_scores = list(results)

    # Track seen section IDs
    seen_ids = set()
    for doc, _ in docs_with_scores:
        seen_ids.add(doc.metadata.get("section_id"))

    # Multi-hop: find "Section X.X" refs in retrieved text
    ref_pattern = re.compile(r"[Ss]ection\s+(\d+\.\d+(?:\.\d+){0,2})")
    follow_up_sections = set()

    for doc, _ in docs_with_scores:
        for m in ref_pattern.finditer(doc.page_content):
            sid = m.group(1)
            if sid not in seen_ids:
                follow_up_sections.add(sid)

    # Also find section numbers mentioned in the original query
    query_sections = set(re.findall(r"\b(\d+\.\d+(?:\.\d+)?)\b", original_query))
    for sid in query_sections:
        if sid not in seen_ids:
            follow_up_sections.add(sid)

    hop_count = 0
    for sid in list(follow_up_sections)[:4]:  # cap at 4 extra hops
        hop_query = f"section {sid} {original_query}"
        hop_results = db.similarity_search_with_relevance_scores(hop_query, k=2)
        for doc, score in hop_results:
            if doc.metadata.get("section_id") not in seen_ids:
                doc.metadata["hop"] = True
                docs_with_scores.append((doc, score))
                seen_ids.add(doc.metadata.get("section_id"))
                hop_count += 1

    return docs_with_scores, hop_count


def build_context(docs_with_scores: list) -> str:
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        meta = doc.metadata
        section = meta.get("section_id", "N/A")
        page = meta.get("page", "N/A")
        hop_note = " [cross-reference]" if meta.get("hop") else ""
        parts.append(
            f"--- Chunk {i} [Section {section}, p.{page}, score={round(score, 3)}]{hop_note} ---\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def call_groq(context: str, question: str) -> str:
    user_text = f"Context (extracted text):\n{context}\n\nQuestion: {question}"
    try:
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


QUERIES = [
    # (tier, query_text)
    ("TIER 1", "What is the purpose of the NASA Systems Engineering Handbook as stated in Section 1.1?"),
    ("TIER 1", "What are all the entry criteria and exit criteria for Phase B: Preliminary Design and Technology Completion?"),
    ("TIER 1", "What is the definition of Product Validation and how does it differ from Product Verification?"),
    ("TIER 1", "List all 17 SE processes defined in the Systems Engineering Engine."),
    ("TIER 1", "What is the purpose and expected result of a System Requirements Review (SRR) according to Table 6.7.1?"),
    ("TIER 2", "How does Technical Risk Management (Section 6.4) feed into the Decision Analysis process (Section 6.8)?"),
    ("TIER 2", "Which SE processes from Chapter 4 are active during Phase C: Final Design and Fabrication?"),
    ("TIER 2", "How do stakeholder expectations flow from Section 4.1 into technical requirements in Section 4.2 and then into the design solution in Section 4.4?"),
    ("TIER 2", "Which lifecycle reviews are conducted between Phase B and Phase C, and what are their success criteria?"),
    ("TIER 2", "How does Configuration Management (Section 6.5) interact with Interface Management (Section 6.3)?"),
    ("TIER 3", "What is the role of HSI in the SE process?"),
    ("TIER 3", "What happens at a KDP and who makes the decision?"),
    ("TIER 3", "What is the difference between a PDR and an SDR?"),
    ("TIER 4", "What does the Systems Engineering Engine diagram show, and what are the three rings or layers?"),
    ("TIER 4", "Describe the NASA Project Life Cycle process flow shown in Figure 2.2.1."),
    ("TIER 5", "What does the handbook say about the relationship between cost and requirements stability? Give the exact section and paragraph."),
    ("TIER 5", "What is NASA's budget for the Artemis program?"),
    ("TIER 5", "Does the handbook recommend a specific number of risk categories for the risk matrix?"),
    ("TIER 6", "A project is in Phase D. A critical interface has failed verification. Using the handbook, trace the full decision path: what verification process applies, how the failure should be managed as a risk, and what technical review would be triggered?"),
    ("TIER 6", "Section 6.6 covers Technical Data Management. What parent process does it belong to, and how does the data it manages get used as inputs in Section 6.7 Technical Assessment?"),
]

OUTPUT_FILE = "/Users/namanvikram/Downloads/Manual QA System/test_results2.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for q_num, (tier, query) in enumerate(QUERIES, 1):
        print(f"\nProcessing Q{q_num} [{tier}]...")

        expanded = expand_acronyms(query)
        docs_with_scores, hop_count = retrieve_and_multihop(expanded, query)
        context = build_context(docs_with_scores)
        answer = call_groq(context, expanded)

        # Gather metadata
        sections = []
        pages = []
        chapters = set()
        for doc, score in docs_with_scores:
            meta = doc.metadata
            sid = meta.get("section_id", "N/A")
            page = meta.get("page", "N/A")
            if sid not in sections:
                sections.append(sid)
            if page not in pages:
                pages.append(page)
            # Chapter = first number before the dot
            if sid and sid != "N/A":
                chap = sid.split(".")[0]
                chapters.add(chap)

        cross = "Y" if len(chapters) > 1 else "N"

        # Count table chunks
        table_chunks = sum(
            1 for doc, _ in docs_with_scores
            if doc.metadata.get("content_type") == "table"
        )

        sep = "=" * 80
        line = "-" * 80
        out.write(f"{sep}\n")
        out.write(f"Q{q_num} [{tier}] - {query}\n")
        out.write(f"RETRIEVED: sections={sections} pages={pages} chapters={sorted(chapters)} cross={cross} hops={hop_count}\n")
        out.write(f"TABLE CHUNKS: {table_chunks}\n")
        out.write(f"{line}\n")
        out.write(f"ANSWER:\n{answer}\n")
        out.write(f"{sep}\n\n")

        print(f"  Done Q{q_num}. Sections={sections[:3]}... Pages={pages[:3]}... Cross={cross} Hops={hop_count}")

print(f"\nAll tests complete. Results saved to: {OUTPUT_FILE}")
