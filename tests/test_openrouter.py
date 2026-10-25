"""HTTP chegarasida taqlid: haqiqiy kalit va pullik so'rov ishlatilmaydi."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from honda_rag import config as c
from honda_rag.embeddings import Embedder
from honda_rag.llm import OpenRouterLLM
from honda_rag.openrouter import OpenRouterClient
from honda_rag.pipeline import RAG
from honda_rag.retrieval import build_index, fingerprint, Retriever


def response(data, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode("utf-8")
    result.headers.update(headers or {})
    return result


class HTTPFixture:
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-secret"}))
        self.stack.enter_context(patch.object(c, "CHAT_MODEL", "openai/gpt-6-luna"))
        self.stack.enter_context(patch.object(c, "EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"))
        self.stack.enter_context(patch.object(c, "EMBEDDING_DIMENSIONS", 0))
        self.session = Mock()
        self.session.headers = {}
        self.session.get.return_value = response({"data": {"is_management_key": False,
                                                           "is_provisioning_key": False}})
        self.stack.enter_context(patch("honda_rag.openrouter.requests.Session", return_value=self.session))
        self.stack.enter_context(patch("honda_rag.openrouter.time.sleep"))


class HTTPTests(HTTPFixture, unittest.TestCase):
    def test_missing_key_fails_before_http(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, r"\.env"):
                OpenRouterClient()
        self.session.post.assert_not_called()

    def test_auth_and_endpoint(self):
        self.session.post.return_value = response({"data": []})
        client = OpenRouterClient()
        client.post("embeddings", {"model": "test-model"})
        self.assertEqual(self.session.headers["Authorization"], "Bearer test-only-secret")
        self.assertEqual(self.session.post.call_args.args[0], "https://openrouter.ai/api/v1/embeddings")

    def test_management_key_is_rejected_before_inference(self):
        for flag in ("is_management_key", "is_provisioning_key"):
            with self.subTest(flag=flag):
                self.session.get.return_value = response({"data": {flag: True, "label": "test-only-secret"}})
                with self.assertRaisesRegex(RuntimeError, "Management/Provisioning") as caught:
                    OpenRouterClient().validate_api_key()
                self.assertIn("settings/keys", str(caught.exception))
                self.assertNotIn("test-only-secret", str(caught.exception))
        self.session.post.assert_not_called()

    def test_normal_key_check_returns_no_private_account_metadata(self):
        self.session.get.return_value = response({"data": {"is_management_key": False,
            "label": "test-only-secret", "limit_remaining": 123, "creator_user_id": "private-user"}})
        self.assertEqual(OpenRouterClient().validate_api_key(), {"authenticated": True, "key_type": "inference"})
        self.session.post.assert_not_called()

    def test_401_with_valid_management_key_explains_actual_cause(self):
        self.session.post.return_value = response({"error": {"message": "User not found."}}, 401)
        self.session.get.return_value = response({"data": {"is_management_key": True}})
        with self.assertRaisesRegex(RuntimeError, "Management/Provisioning"):
            OpenRouterClient().post("embeddings", {})
        self.assertEqual(self.session.post.call_count, 1)

    def test_invalid_key_check_is_actionable_without_echoing_response(self):
        self.session.get.return_value = response({"error": {"message": "test-only-secret"}}, 401)
        with self.assertRaisesRegex(RuntimeError, "tanilmadi") as caught:
            OpenRouterClient().validate_api_key()
        self.assertNotIn("test-only-secret", str(caught.exception))

    def test_401_message_does_not_echo_secret(self):
        self.session.post.return_value = response({"error": "test-only-secret"}, 401)
        with self.assertRaises(RuntimeError) as caught:
            OpenRouterClient().post("embeddings", {})
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-only-secret", str(caught.exception))
        self.assertEqual(self.session.post.call_count, 1)

    def test_rate_limit_retries_then_succeeds(self):
        self.session.post.side_effect = [response({}, 429, {"Retry-After": "0"}), response({"data": []})]
        self.assertEqual(OpenRouterClient().post("embeddings", {}), {"data": []})
        self.assertEqual(self.session.post.call_count, 2)

    def test_timeout_is_not_automatically_billed_again(self):
        self.session.post.side_effect = requests.Timeout("test-only-secret")
        with self.assertRaisesRegex(RuntimeError, "vaqti tugadi") as caught:
            OpenRouterClient().post("chat/completions", {})
        self.assertNotIn("test-only-secret", str(caught.exception))
        self.assertEqual(self.session.post.call_count, 1)

    def test_embeddings_preserve_order_and_normalize(self):
        self.session.post.return_value = response({"data": [
            {"index": 1, "embedding": [0, 4]}, {"index": 0, "embedding": [3, 0]}]})
        with Embedder() as embedder:
            vectors = embedder.encode(["first", "second"])
        np.testing.assert_allclose(vectors, [[1, 0], [0, 1]])
        self.assertEqual(self.session.post.call_args.kwargs["json"]["input"], ["first", "second"])
        self.session.close.assert_called_once()

    def test_query_instruction_only_on_qwen_queries(self):
        self.session.post.return_value = response({"data": [{"index": 0, "embedding": [1, 0]}]})
        with Embedder() as embedder:
            embedder.encode(["Moy haqida savol?"], query=True)
        sent = self.session.post.call_args.kwargs["json"]["input"][0]
        self.assertTrue(sent.startswith("Instruct:"))
        self.assertIn("Query:Moy haqida savol?", sent)

    def test_non_qwen_model_has_no_qwen_prefix(self):
        self.session.post.return_value = response({"data": [{"index": 0, "embedding": [1, 0]}]})
        with patch.object(c, "EMBEDDING_MODEL", "openai/text-embedding-3-small"):
            with Embedder() as embedder:
                embedder.encode(["question"], query=True)
        self.assertEqual(self.session.post.call_args.kwargs["json"]["input"], ["question"])

    def test_duplicate_or_missing_embedding_indices_are_rejected(self):
        self.session.post.return_value = response({"data": [
            {"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}]})
        with Embedder() as embedder:
            with self.assertRaisesRegex(RuntimeError, "index"):
                embedder.encode(["first", "second"])

    def test_nan_and_zero_embeddings_are_rejected(self):
        for vector in ([float("nan"), 1], [0, 0]):
            with self.subTest(vector=vector):
                self.session.post.return_value = response({"data": [{"index": 0, "embedding": vector}]})
                with Embedder() as embedder:
                    with self.assertRaises(RuntimeError):
                        embedder.encode(["first"])

    def test_batch_dimension_changes_are_rejected(self):
        self.session.post.side_effect = [
            response({"data": [{"index": 0, "embedding": [1, 0]}]}),
            response({"data": [{"index": 0, "embedding": [1, 0, 0]}]})]
        with Embedder() as embedder:
            with self.assertRaisesRegex(RuntimeError, "o'lchami"):
                embedder.encode(["first", "second"], batch_size=1)

    def test_requested_dimensions_are_sent_and_checked(self):
        self.session.post.return_value = response({"data": [{"index": 0, "embedding": [1, 0]}]})
        with patch.object(c, "EMBEDDING_DIMENSIONS", 3):
            with Embedder() as embedder:
                with self.assertRaisesRegex(RuntimeError, "o'lchami"):
                    embedder.encode(["first"])
        self.assertEqual(self.session.post.call_args.kwargs["json"]["dimensions"], 3)

    def test_chat_targets_luna_and_uses_strict_schema(self):
        answer = {"supported": True, "answer": "0W-20 tavsiya etiladi. [1]", "sources": [1]}
        self.session.post.return_value = response({"choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]})
        model = OpenRouterLLM()
        result = model.answer("Qanday moy?", [{"reference": 1, "pdf_page": 664, "text": "0W-20"}])
        payload = self.session.post.call_args.kwargs["json"]
        self.assertEqual(result, answer)
        self.assertEqual(payload["model"], "openai/gpt-6-luna")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertNotIn("temperature", payload)
        self.assertIn("Uzbek", payload["messages"][0]["content"])

    def test_truncated_or_bad_chat_output_is_not_a_grounded_answer(self):
        cases = [("length", '{"supported":'), ("stop", 'not json'),
                 ("stop", '{"supported":"true","answer":"x","sources":[1]}')]
        for finish, content in cases:
            with self.subTest(finish=finish, content=content):
                self.session.post.return_value = response({"choices": [
                    {"finish_reason": finish, "message": {"content": content}}]})
                with self.assertRaises(RuntimeError):
                    OpenRouterLLM().answer("question", [{"reference": 1, "pdf_page": 1, "text": "text"}])


class IndexMigrationTests(HTTPFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.stack.enter_context(patch.object(c, "INDEX", root / "index" / "openrouter"))
        self.stack.enter_context(patch.object(c, "DOCUMENT_CACHE", root / "cache"))
        self.stack.enter_context(patch.object(c, "PDF", root / "manual.pdf"))
        c.PDF.write_bytes(b"%PDF-test fixture")
        c.DOCUMENT_CACHE.mkdir()
        pages = [{"pdf_page": 664, "book_page": 662, "text": "Honda motor oil recommendation is 0W-20."},
                 {"pdf_page": 32, "book_page": 30, "text": "Fuel tank capacity is 46.9 L."}]
        chunks = [x | {"id": f"p{x['pdf_page']:04d}-0000", "source": c.PDF.name, "title": c.TITLE} for x in pages]
        metadata = {"pdf_sha256": fingerprint(c.PDF), "chunk_words": c.CHUNK_WORDS,
                    "chunk_overlap": c.CHUNK_OVERLAP}
        for name, data in (("metadata.json", metadata), ("chunks.json", chunks), ("pages.json", pages)):
            (c.DOCUMENT_CACHE / name).write_text(json.dumps(data), encoding="utf-8")
        self.session.post.side_effect = self.route_request

    def route_request(self, url, **kwargs):
        payload = kwargs["json"]
        if url.endswith("/embeddings"):
            items = [{"index": i, "embedding": ([0, 4] if "46.9" in text else [3, 0])}
                     for i, text in enumerate(payload["input"])]
            return response({"data": list(reversed(items))})
        return response({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "supported": True, "answer": "0W-20 moyi tavsiya etiladi. [1]", "sources": [1]})}}]})

    def test_cached_text_is_reembedded_and_rag_uses_real_source_page(self):
        with patch("honda_rag.documents.extract_document", side_effect=AssertionError("cache should be used")):
            metadata = build_index()
        self.assertEqual(metadata["embedding_model"], c.EMBEDDING_MODEL)
        self.assertEqual(metadata["dimensions"], 2)
        with RAG() as rag:
            result = rag.ask("Qanday motor moyi tavsiya etiladi?")
        self.assertTrue(result["supported"])
        self.assertEqual(result["sources"][0]["pdf_page"], 664)
        self.assertEqual(result["retrieved"][0]["search_query"], result["question"])
        endpoints = [call.args[0].rsplit("/", 1)[-1] for call in self.session.post.call_args_list]
        self.assertEqual(endpoints, ["embeddings", "embeddings", "completions"])

    def test_different_embedding_model_rejects_existing_index(self):
        build_index()
        with patch.object(c, "EMBEDDING_MODEL", "openai/text-embedding-3-small"):
            with self.assertRaisesRegex(RuntimeError, "o'zgargan"):
                Retriever()
        self.assertEqual(self.session.post.call_count, 1)

    def test_dimension_setting_change_rejects_existing_index(self):
        build_index()
        with patch.object(c, "EMBEDDING_DIMENSIONS", 1):
            with self.assertRaisesRegex(RuntimeError, "o'zgargan"):
                Retriever()

    def test_chat_model_change_does_not_reembed_the_book(self):
        build_index()
        with patch.object(c, "CHAT_MODEL", "another/chat-model"):
            with Retriever() as retriever:
                self.assertEqual(len(retriever.chunks), 2)
        self.assertEqual(self.session.post.call_count, 1)

    def test_unchanged_index_is_reused_without_api_call(self):
        build_index()
        self.session.post.reset_mock()
        build_index()
        self.session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
