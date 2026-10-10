import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm import GroqConfigurationError, generate_answer


class LlmTests(unittest.TestCase):
    def test_generate_answer_uses_configured_model_and_returns_text(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))]
        )
        create_completion = Mock(return_value=completion)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            )
        )

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "test-model"}),
            patch("llm.get_client", return_value=client),
        ):
            self.assertEqual(generate_answer("question", "context"), "Answer")

        create_completion.assert_called_once()
        self.assertEqual(create_completion.call_args.kwargs["model"], "test-model")

    def test_generate_answer_rejects_missing_key(self):
        with (
            patch.dict(os.environ, {"GROQ_MODEL": "test-model"}, clear=True),
            self.assertRaisesRegex(GroqConfigurationError, "GROQ_API_KEY"),
        ):
            generate_answer("question", "")

    def test_generate_answer_rejects_placeholder_model(self):
        with (
            patch.dict(
                os.environ,
                {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "your_groq_model"},
            ),
            self.assertRaisesRegex(GroqConfigurationError, "placeholder"),
        ):
            generate_answer("question", "")

    def test_generate_answer_rejects_placeholder_key(self):
        with (
            patch.dict(
                os.environ,
                {"GROQ_API_KEY": "your_groq_api_key", "GROQ_MODEL": "test-model"},
            ),
            self.assertRaisesRegex(GroqConfigurationError, "GROQ_API_KEY.*placeholder"),
        ):
            generate_answer("question", "")


if __name__ == "__main__":
    unittest.main()
