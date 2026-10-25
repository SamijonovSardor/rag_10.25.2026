import unittest
from unittest.mock import Mock

from honda_rag.documents import chunk_page
from honda_rag.pipeline import RAG


class EvidenceTests(unittest.TestCase):
    def make_rag(self, generated):
        rag = object.__new__(RAG)
        rag.retriever = Mock()
        rag.retriever.search.return_value = [{"reference": 1, "pdf_page": 620, "text": "source"}]
        rag.llm = Mock()
        rag.llm.answer.return_value = generated
        return rag

    def test_fabricated_reference_is_not_presented_as_grounded_answer(self):
        rag = self.make_rag({"supported": True, "answer": "False claim [9]", "sources": [9]})
        self.assertFalse(rag.ask("question")["supported"])
        self.assertEqual(rag.ask("question")["sources"], [])

    def test_inline_reference_must_match_source_ids(self):
        rag = self.make_rag({"supported": True, "answer": "Claim [2]", "sources": [1]})
        self.assertFalse(rag.ask("question")["supported"])

    def test_abstention_does_not_show_search_results_as_used_sources(self):
        rag = self.make_rag({"supported": False, "answer": "Unknown", "sources": []})
        result = rag.ask("question")
        self.assertFalse(result["supported"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(len(result["retrieved"]), 1)

    def test_grounded_answer_keeps_actual_pdf_page(self):
        rag = self.make_rag({"supported": True, "answer": "Known fact [1]", "sources": [1]})
        self.assertEqual(rag.ask("question")["sources"][0]["pdf_page"], 620)

    def test_valid_source_list_without_inline_marker_is_formatted(self):
        rag = self.make_rag({"supported": True, "answer": "Known fact", "sources": [1]})
        result = rag.ask("question")
        self.assertTrue(result["supported"])
        self.assertIn("[1]", result["answer"])


class PageBoundaryTests(unittest.TestCase):
    def test_chunk_overlap_preserves_page_provenance(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_page(text, 42, 40)
        self.assertEqual({x["pdf_page"] for x in chunks}, {42})
        self.assertEqual({x["book_page"] for x in chunks}, {40})
        self.assertEqual(chunks[0]["text"].split()[-35:], chunks[1]["text"].split()[:35])
        self.assertIn("word499", chunks[-1]["text"])


if __name__ == "__main__":
    unittest.main()
