"""
Tests for the Q&A Pipeline.

Run: pytest tests/ -v
"""

import os
import pytest
from transformers import pipeline as hf_pipeline

from src.knowledge_base import build_knowledge_base
from src.pipeline import ask_question, get_llm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all tests."""
    return build_knowledge_base(DATA_DIR)


@pytest.fixture(scope="module")
def llm():
    """Load the LLM once for all tests."""
    return get_llm()


# ────────────────────────────────
# ask_question return structure
# ────────────────────────────────
class TestAskQuestionStructure:
    def test_returns_dict(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result, dict), "ask_question should return a dict"

    def test_has_answer_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "answer" in result, "Result dict must have an 'answer' key"

    def test_has_sources_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "sources" in result, "Result dict must have a 'sources' key"

    def test_answer_is_string(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["answer"], str), "'answer' should be a string"
        assert len(result["answer"].strip()) > 0, "'answer' should not be empty"

    def test_sources_is_list(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["sources"], list), "'sources' should be a list"
        assert len(result["sources"]) > 0, "'sources' should not be empty"


# ────────────────────────────────
# Retrieval quality
# ────────────────────────────────
class TestRetrieval:
    def test_retrieves_pricing_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much does the Growth package cost?")
        sources_text = " ".join(result["sources"]).lower()
        assert "growth" in sources_text or "$5,500" in sources_text, (
            "Sources should contain pricing-related content"
        )

    def test_retrieves_seo_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Do you offer SEO services?")
        sources_text = " ".join(result["sources"]).lower()
        assert "seo" in sources_text or "keyword" in sources_text, (
            "Sources should contain SEO-related content"
        )

    def test_different_questions_get_different_sources(self, vector_store, llm):
        r1 = ask_question(vector_store, llm, "How does onboarding work?")
        r2 = ask_question(vector_store, llm, "What are your PPC management fees?")
        assert r1["sources"] != r2["sources"], (
            "Different questions should retrieve different chunks"
        )


# ────────────────────────────────
# Answer generation
# ────────────────────────────────
class TestAnswerGeneration:
    def test_answer_is_not_just_the_prompt(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Can I cancel my contract?")
        assert "Context:" not in result["answer"], (
            "Answer should be the generated text, not the full prompt"
        )

    def test_answer_responds_to_question(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much is the Starter package?")
        answer = result["answer"].lower()
        assert "2,500" in answer or "2500" in answer or "starter" in answer, (
            "Answer should address the pricing question"
        )


# ────────────────────────────────
# Bonus: edge cases and extra coverage
# ────────────────────────────────
class TestBonus:
    def test_empty_question_returns_prompt_message(self, vector_store, llm):
        result = ask_question(vector_store, llm, "   ")
        assert result["sources"] == [], "Empty input should retrieve no sources"
        assert "question" in result["answer"].lower(), (
            "Empty input should return a user-facing prompt"
        )

    def test_sources_have_exactly_three_chunks(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What is the refund policy?")
        assert len(result["sources"]) == 3, "Retriever should return top-3 chunks"

    def test_unknown_topic_falls_back_gracefully(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What is the airspeed velocity of a swallow?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0, "Answer must still be a non-empty string"

    def test_sources_are_all_strings(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Do you offer content marketing?")
        assert all(isinstance(s, str) for s in result["sources"]), (
            "Every source entry should be a string"
        )
