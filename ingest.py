"""
ingest.py — One-time ingestion pipeline for the NASA Systems Engineering Handbook.

Steps:
  1. Load the PDF with page-level metadata (PyPDFLoader)
  2. Detect section boundaries via regex on heading patterns
  3. Split into section-aware chunks (with RecursiveCharacterTextSplitter fallback
     for oversized sections)
  4. Tag each chunk with section_id, chapter, page, source
  5. Embed chunks and persist to ChromaDB

Run once:
    python ingest.py
"""

import os
import re
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_PATH = os.getenv("PDF_PATH", "nasa_systems_engineering_handbook_0.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nasa_se_handbook")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Matches section headings like "1.1 Purpose", "6.3.2.1 Title"
# - Must be at start of line OR immediately after a digit (PDF text runs together)
# - Section number must be 1-4 levels deep (e.g. 1.0, 2.3, 4.5.1, 6.3.2.1)
# - Chapter number must be 1-9 (avoids matching page numbers like 182.0)
# - Followed by a space+word OR directly a capital letter (no-space headings)
SECTION_PATTERN = re.compile(
    r"(?:^|(?<=\n))([1-9]\.\d{1,2}(?:\.\d{1,2}){0,2})(?:\s+[A-Z]|\s*\n)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_embedding_function():
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"Using HuggingFace embeddings ({EMBEDDING_MODEL})...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def extract_section_id(text: str) -> str:
    """Return the first section number found in text, e.g. '6.3.2'."""
    m = SECTION_PATTERN.search(text)
    return m.group(1) if m else ""


def extract_chapter(section_id: str) -> int:
    """Return the top-level chapter number from a section id like '6.3.2'."""
    if not section_id:
        return 0
    try:
        return int(section_id.split(".")[0])
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_pdf(path: str) -> list[Document]:
    print(f"Loading PDF: {path}")
    if not os.path.exists(path):
        sys.exit(f"ERROR: PDF not found at '{path}'. Download it and place it in the project root.")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages.")
    return pages


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def chunk_by_section(pages: list[Document]) -> list[Document]:
    """
    Concatenate all page text into one stream, then split at section boundaries
    detected by SECTION_PATTERN.  Each resulting chunk inherits the page number
    of the first page it came from.

    Oversized sections (> CHUNK_SIZE chars) are further split with
    RecursiveCharacterTextSplitter to stay within context limits.
    """

    # Build a flat list of (page_number, text) pairs
    page_texts: list[tuple[int, str]] = []
    for doc in pages:
        page_num = doc.metadata.get("page", 0) + 1  # PyPDF uses 0-indexed pages
        page_texts.append((page_num, doc.page_content))

    # Concatenate — we need to track page boundaries as we go
    full_text = ""
    # Map from character offset → page number
    page_offsets: list[tuple[int, int]] = []  # (start_char, page_num)
    for page_num, text in page_texts:
        page_offsets.append((len(full_text), page_num))
        full_text += text + "\n"

    def char_to_page(char_idx: int) -> int:
        """Binary-search the page_offsets list to find page for a char position."""
        lo, hi = 0, len(page_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if page_offsets[mid][0] <= char_idx:
                lo = mid
            else:
                hi = mid - 1
        return page_offsets[lo][1]

    # Find all section boundary positions
    boundaries = [m.start() for m in SECTION_PATTERN.finditer(full_text)]
    boundaries.append(len(full_text))  # sentinel

    print(f"  Detected {len(boundaries) - 1} section boundaries.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        section_text = full_text[start:end].strip()
        if not section_text:
            continue

        page = char_to_page(start)
        section_id = extract_section_id(section_text)
        chapter = extract_chapter(section_id)

        # Parent section = section_id stripped to 2 levels (e.g. "6.3.2" → "6.3")
        parts = section_id.split(".") if section_id else []
        parent_section = ".".join(parts[:2]) if len(parts) >= 2 else section_id

        base_metadata = {
            "section_id": section_id,
            "parent_section": parent_section,
            "chapter": chapter,
            "page": page,
            "source": PDF_PATH,
        }

        if len(section_text) <= CHUNK_SIZE:
            chunks.append(Document(page_content=section_text, metadata=base_metadata))
        else:
            # Split oversized section into sub-chunks, all inheriting same metadata
            sub_docs = splitter.create_documents([section_text], metadatas=[base_metadata])
            chunks.extend(sub_docs)

    print(f"  Created {len(chunks)} chunks (avg {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars).")
    return chunks


# ---------------------------------------------------------------------------
# Table extraction (pdfplumber)
# ---------------------------------------------------------------------------


def extract_tables(path: str) -> list[Document]:
    """
    Extract all structured tables from the PDF using pdfplumber.
    Each table row is stored as a Document with column headers prepended,
    so every row chunk is self-contained and queryable without needing
    the header page.

    Multi-page tables (same column headers on consecutive pages) are
    detected and merged into a single logical table.
    """
    try:
        import pdfplumber
    except ImportError:
        print("  pdfplumber not installed — skipping table extraction.")
        return []

    table_chunks: list[Document] = []
    prev_headers: list[str] | None = None
    prev_table_name = ""
    prev_page = 0

    with pdfplumber.open(path) as pdf:
        for page_obj in pdf.pages:
            page_num = page_obj.page_number  # 1-indexed
            tables = page_obj.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Skip header/footer tables (1 row or <3 cols)
                if len(table) < 2 or len(table[0]) < 3:
                    continue

                headers = [str(c).strip().replace("\n", " ") if c else "" for c in table[0]]

                # Detect multi-page continuation: same headers on adjacent page
                is_continuation = (
                    prev_headers == headers
                    and page_num <= prev_page + 2
                )

                # Try to find a table caption in the page text
                page_text = page_obj.extract_text() or ""
                table_name_match = re.search(r"TABLE\s+[\d\.\-]+[^\n]*", page_text)
                table_name = table_name_match.group(0).strip() if table_name_match else prev_table_name

                if not is_continuation:
                    prev_table_name = table_name

                data_rows = table[1:]  # skip header row
                for row in data_rows:
                    if not any(cell for cell in row):
                        continue
                    cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]

                    # Build self-contained row text: "Header: Value | Header: Value ..."
                    row_text = f"[{table_name or 'Table'}]\n"
                    row_text += " | ".join(
                        f"{h}: {v}" for h, v in zip(headers, cells) if h or v
                    )

                    # Extract section_id from table name or page text
                    section_id = extract_section_id(page_text)
                    parts = section_id.split(".") if section_id else []
                    parent_section = ".".join(parts[:2]) if len(parts) >= 2 else section_id
                    chapter = extract_chapter(section_id)

                    table_chunks.append(Document(
                        page_content=row_text,
                        metadata={
                            "section_id": section_id,
                            "parent_section": parent_section,
                            "chapter": chapter,
                            "page": page_num,
                            "source": path,
                            "content_type": "table",
                            "table_name": table_name,
                        }
                    ))

                prev_headers = headers
                prev_page = page_num

    print(f"  Extracted {len(table_chunks)} table row chunks from {path}.")
    return table_chunks


# ---------------------------------------------------------------------------
# Embed + Store
# ---------------------------------------------------------------------------


def embed_and_store(chunks: list[Document], embedding_fn) -> None:
    from langchain_chroma import Chroma

    print(f"Embedding {len(chunks)} chunks and saving to {CHROMA_DIR}...")

    # Remove existing collection to avoid duplicates on re-run
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("  Cleared existing vector store.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"  Vector store saved to {CHROMA_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    pages = load_pdf(PDF_PATH)
    chunks = chunk_by_section(pages)
    table_chunks = extract_tables(PDF_PATH)
    all_chunks = chunks + table_chunks
    print(f"  Total chunks: {len(all_chunks)} ({len(chunks)} text + {len(table_chunks)} table rows)")
    embedding_fn = get_embedding_function()
    embed_and_store(all_chunks, embedding_fn)
    print("\nIngestion complete. Run 'python app.py' to start the chatbot.")


if __name__ == "__main__":
    main()
