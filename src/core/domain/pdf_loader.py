from PyPDF2 import PdfReader
from loguru import logger


class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pages = []

    def load_pdf(self):
        reader = PdfReader(self.pdf_path)
        num_pages = len(reader.pages)
        print(f"Total pages in PDF: {num_pages}")

        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                self.pages.append(text)
            else:
                self.pages.append("")  # Handle empty pages
            if (i + 1) % 100 == 0:
                logger.debug(f"Loaded {i + 1} pages")
        logger.debug("PDF loading completed.")
