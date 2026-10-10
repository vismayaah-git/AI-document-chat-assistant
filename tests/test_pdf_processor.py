import unittest
from unittest.mock import Mock, patch

from pdf_processor import extract_text_from_pdf, split_text


class PdfProcessorTests(unittest.TestCase):
    @patch("pdf_processor.PdfReader")
    def test_extract_text_joins_pages_and_handles_empty_pages(self, reader_class):
        reader_class.return_value.pages = [
            Mock(extract_text=Mock(return_value="First page")),
            Mock(extract_text=Mock(return_value=None)),
            Mock(extract_text=Mock(return_value="Third page")),
        ]

        self.assertEqual(extract_text_from_pdf(object()), "First page\n\nThird page")

    def test_split_text_returns_no_chunks_for_blank_input(self):
        self.assertEqual(split_text(" \n\t "), [])

    def test_split_text_splits_long_text(self):
        chunks = split_text("word " * 300)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
