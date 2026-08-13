"""
Document Q&A Pipeline.

Retrieves the most relevant chunks from a FAISS vector store, feeds them
to a local flan-t5-base LLM through a prompt template, and exposes both
an interactive REPL and a one-shot `--query` mode.
"""

import argparse
import os
import sys
from typing import Callable, Dict, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


LLMCallable = Callable[[str], List[Dict[str, str]]]


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> LLMCallable:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> List[Dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm: LLMCallable, question: str) -> Dict[str, object]:
    """Retrieve the top-k chunks for `question` and generate an answer.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    docs = vector_store.similarity_search(question, k=3)
    sources: List[str] = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = llm(prompt)
    answer = result[0]["generated_text"].strip()

    return {"answer": answer, "sources": sources}


def _print_result(result: Dict[str, object]) -> None:
    """Pretty-print a result dict from ask_question()."""
    print("\n📄 Sources:")
    for i, src in enumerate(result["sources"], start=1):
        snippet = src.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        print(f"  {i}. {snippet}")
    print(f"\n💬 Answer: {result['answer']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop with optional --query one-shot mode."""
    parser = argparse.ArgumentParser(description="Marketing-agency Q&A chatbot.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    if args.query is not None:
        result = ask_question(vector_store, llm, args.query)
        _print_result(result)
        return

    print("Ask me anything about the agency. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = ask_question(vector_store, llm, question)
            _print_result(result)
        except Exception as e:
            print(f"Something went wrong: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
