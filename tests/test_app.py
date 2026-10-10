import unittest
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_textless_pdf_shows_ocr_message_without_crashing(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertFalse(app.exception)

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf_buffer = BytesIO()
        writer.write(pdf_buffer)

        app.file_uploader[0].upload(
            "blank.pdf", pdf_buffer.getvalue(), "application/pdf"
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("No selectable text" in warning.value for warning in app.warning)
        )


if __name__ == "__main__":
    unittest.main()
