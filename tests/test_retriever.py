import unittest
from types import SimpleNamespace
from unittest.mock import patch

from retriever import create_bm25, hybrid_search, rerank


class FakeVectorStore:
    def similarity_search(self, query, k=3):
        self.last_query = query
        self.last_k = k
        return [SimpleNamespace(page_content="Semantic result")]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            "Machine learning finds patterns",
            "Gardening needs sunlight",
            "Machine translation uses models",
        ]

    def test_hybrid_search_matches_keywords_case_insensitively(self):
        vector_store = FakeVectorStore()
        results, keyword_results = hybrid_search(
            self.chunks, vector_store, create_bm25(self.chunks), "MACHINE"
        )

        self.assertEqual(results[0].page_content, "Semantic result")
        self.assertEqual(vector_store.last_query, "MACHINE")
        self.assertEqual(vector_store.last_k, 3)
        self.assertEqual(set(keyword_results), {self.chunks[0], self.chunks[2]})

    def test_hybrid_search_does_not_return_nonmatching_keywords(self):
        _, keyword_results = hybrid_search(
            self.chunks, FakeVectorStore(), create_bm25(self.chunks), "astronomy"
        )

        self.assertEqual(keyword_results, [])

    def test_rerank_returns_empty_for_no_candidates(self):
        with patch("retriever.create_reranker") as create_reranker:
            self.assertEqual(rerank("question", []), [])

        create_reranker.assert_not_called()

    def test_rerank_orders_candidates_by_model_score(self):
        model = SimpleNamespace(predict=lambda pairs: [0.1, 0.9][: len(pairs)])
        with patch("retriever.create_reranker", return_value=model):
            results = rerank("question", ["first", "second"])

        self.assertEqual(results, ["second", "first"])


if __name__ == "__main__":
    unittest.main()
