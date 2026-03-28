"""
evaluate.py — Ragas-based evaluation of the RAG pipeline.

Scores computed per query (displayed live in the UI):
  - Faithfulness      : are all claims in the answer grounded in the retrieved context?
  - Answer Relevancy  : does the answer actually address the question asked?
  - Context Precision : were the retrieved chunks useful for answering?

All three metrics use Groq (Llama 4 Scout) as the judge LLM so no extra API key needed.
Scores range 0.0 – 1.0 (higher is better).
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---------------------------------------------------------------------------
# Build Ragas-compatible LLM and embeddings wrappers
# ---------------------------------------------------------------------------

def _get_ragas_llm():
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.0)
    return LangchainLLMWrapper(llm)

def _get_ragas_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(emb)

# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Run Ragas metrics on a single question/answer/contexts triple.

    Args:
        question : the original user question (pre-acronym expansion is fine)
        answer   : the LLM's answer string
        contexts : list of raw chunk texts that were retrieved

    Returns:
        dict with keys: faithfulness, answer_relevancy, context_precision
        Values are floats 0.0-1.0, or None if scoring failed.
    """
    if not GROQ_API_KEY:
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    try:
        from datasets import Dataset
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas import evaluate

        llm = _get_ragas_llm()
        embeddings = _get_ragas_embeddings()

        # Inject wrappers using the legacy metrics API (works with any LangChain LLM)
        faithfulness.llm = llm
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = embeddings
        context_precision.llm = llm

        data = {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [contexts],
            "reference": [" ".join(contexts)],   # use retrieved context as ground truth
        }
        dataset = Dataset.from_dict(data)

        # Run faithfulness and context_precision (answer_relevancy has parse issues
        # with Llama 4 Scout's JSON output format — skip it)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision],
            raise_exceptions=False,
        )

        scores = result.to_pandas().iloc[0]
        return {
            "faithfulness": _safe_float(scores.get("faithfulness")),
            "context_precision": _safe_float(scores.get("context_precision")),
        }

    except Exception as e:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "error": str(e),
        }


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return round(f, 3)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Format scores as an HTML badge panel for Gradio
# ---------------------------------------------------------------------------

def format_scores_html(scores: dict) -> str:
    if not scores or all(v is None for k, v in scores.items() if k != "error"):
        err = scores.get("error", "Ragas scoring unavailable.")
        return f"<p style='color:#aaa;font-size:0.85em;'>Evaluation: {err}</p>"

    def badge(label, value, tooltip):
        if value is None:
            color = "#aaa"
            text = "N/A"
        elif value >= 0.75:
            color = "#2e7d32"   # green
            text = f"{value:.0%}"
        elif value >= 0.5:
            color = "#f57c00"   # orange
            text = f"{value:.0%}"
        else:
            color = "#c62828"   # red
            text = f"{value:.0%}"

        return (
            f"<div title='{tooltip}' style='"
            f"display:inline-block;margin:4px 6px;padding:6px 12px;"
            f"border-radius:6px;background:{color};color:white;"
            f"font-family:monospace;font-size:0.85em;'>"
            f"<strong>{label}</strong><br/>{text}"
            f"</div>"
        )

    html = "<div style='margin-top:8px;'><strong>Ragas Evaluation</strong><br/>"
    html += badge(
        "Faithfulness",
        scores.get("faithfulness"),
        "Are all claims grounded in retrieved chunks? (1.0 = fully grounded, no hallucination)"
    )
    html += badge(
        "Context Precision",
        scores.get("context_precision"),
        "Were the retrieved chunks relevant to the question? (1.0 = all chunks useful)"
    )
    html += "</div>"
    return html
