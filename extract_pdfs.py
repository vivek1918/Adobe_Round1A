import fitz  # PyMuPDF
import re
import json
import os
import tempfile
from typing import List, Dict, Optional
from pdf2image import convert_from_path
import pytesseract
import numpy as np
import time
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class EnhancedPDFToOutline:
    def __init__(self, use_ml=False):
        self.title = None
        self.outline = []
        self.current_page = 0
        self.use_ml = use_ml
        self.ml_model = None
        self._initialize_ml_model()

    def _initialize_ml_model(self):
        """Improved ML model using TF-IDF and RandomForest"""
        if self.use_ml:
            headings = [
                "Introduction", "Chapter 1 Introduction", "1.1 Motivation", "2.1 Methods", "Conclusion",
                "Abstract", "Bibliography", "References", "Figure 1: Overview", "Table 2: Results"
            ]
            non_headings = [
                "This is a sample paragraph explaining the methodology.",
                "In recent years, artificial intelligence has grown rapidly.",
                "The experiment was conducted over 10 days.",
                "We thank our supervisors for their help.",
                "This paper focuses on multiple objectives."
            ]
            X = headings + non_headings
            y = [1] * len(headings) + [0] * len(non_headings)

            self.ml_model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            self.ml_model.fit(X, y)

    def _classify_pdf_type(self, doc: fitz.Document) -> str:
        text_pages = sum(1 for page in doc if page.get_text().strip())
        if text_pages == len(doc):
            return "Native PDF"
        elif text_pages == 0:
            return "Scanned PDF"
        else:
            return "Mixed PDF"

    def _extract_title(self, doc: fitz.Document):
        if len(doc) == 0:
            return
        first_page = doc[0]
        text = first_page.get_text("text")

        if "Dissertation" in text and "an der Fakultät" in text:
            candidates = []
            for block in first_page.get_text("blocks"):
                block_text = block[4].strip()
                if not block_text or block_text == "Dissertation":
                    continue
                if any(x in block_text for x in ["an der Fakultät", "vorgelegt von", "vom", "den"]):
                    continue
                font_size = block[3] - block[1]
                candidates.append((font_size, block_text))
            if candidates:
                candidates.sort(reverse=True)
                self.title = candidates[0][1]
                return

        candidates = []
        for block in first_page.get_text("blocks"):
            text = block[4].strip()
            if not text:
                continue
            font_size = block[3] - block[1]
            is_centered = abs((block[0] + block[2]) / 2 - 300) < 50
            is_bold = any(span['flags'] & 16 for span in first_page.get_text("dict")['blocks'][0]['lines'][0]['spans'])

            if len(text) < 100:
                candidates.append({
                    'text': text,
                    'score': font_size + (100 if is_centered else 0) + (50 if is_bold else 0)
                })

        if candidates:
            self.title = max(candidates, key=lambda x: x['score'])['text']

    def convert_pdf(self, pdf_path: str) -> Dict[str, any]:
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pdf_type = self._classify_pdf_type(doc)
            print(f"[PDF Type] {os.path.basename(pdf_path)}: {pdf_type}")

            self._extract_title(doc)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for page_num, page in enumerate(doc, start=1):
                    futures.append(executor.submit(self._process_page, page, page_num))
                for future in concurrent.futures.as_completed(futures):
                    self.outline.extend(future.result())

            self._extract_figures_with_ocr(pdf_path)

            doc.close()
            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")
            return {
                "title": self.title or os.path.basename(pdf_path),
                "outline": self._clean_outline()
            }
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Processing failed after {processing_time:.2f} seconds")
            return {
                "title": os.path.basename(pdf_path),
                "outline": [],
                "error": str(e)
            }

    def _process_page(self, page, page_num):
        blocks = page.get_text("blocks")
        results = []
        for block in blocks:
            text = self._clean_text(block[4])
            if not text:
                continue
            level = self._determine_heading_level(text, block)
            if level:
                results.append({
                    "level": level,
                    "text": text,
                    "page": page_num,
                    "type": "text"
                })
        return results

    def _extract_figures_with_ocr(self, pdf_path: str):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                images = convert_from_path(pdf_path, output_folder=tmpdir, fmt='jpeg', first_page=1, last_page=5)
                for i, img in enumerate(images, start=1):
                    ocr_text = pytesseract.image_to_string(img)
                    self._process_ocr_text(ocr_text, i)
        except Exception as e:
            print(f"OCR processing skipped: {str(e)}")

    def _process_ocr_text(self, text: str, page_num: int):
        figure_pattern = re.compile(r'(Figure\s*\d+|Fig\.\s*\d+)[:\s]*(.*?)(?=\n|$)', re.IGNORECASE)
        table_pattern = re.compile(r'(Table\s*\d+)[:\s]*(.*?)(?=\n|$)', re.IGNORECASE)

        for match in figure_pattern.finditer(text):
            self.outline.append({
                "level": "H4",
                "text": f"{match.group(1)}: {match.group(2).strip()}",
                "page": page_num,
                "type": "figure"
            })

        for match in table_pattern.finditer(text):
            self.outline.append({
                "level": "H4",
                "text": f"{match.group(1)}: {match.group(2).strip()}",
                "page": page_num,
                "type": "table"
            })

    def _clean_text(self, text: str) -> str:
        patterns = [
            r'\.{3,}\s*\d*$', r'^\d+\s*[-.]\s*', r'\s*\d+\s*$', r'[\x00-\x1F\x7F-\x9F]'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return ' '.join(text.split()).strip()

    def _determine_heading_level(self, text: str, block) -> Optional[str]:
        if self.use_ml and self.ml_model:
            prediction = self.ml_model.predict([text])[0]
            if prediction == 1:
                return self._determine_heading_level_rules(text, block)
        return self._determine_heading_level_rules(text, block)

    def _determine_heading_level_rules(self, text: str, block) -> Optional[str]:
        if len(text) > 200 or self._is_signature_or_date(text) or self._is_document_metadata(text):
            return None

        patterns = {
            'H1': [r'^[A-Z][A-Za-z\s]{10,}$', r'^[IVX]+\.?\s+[A-Z][a-z]', r'^Abstract\b', r'^References?\b'],
            'H2': [r'^[1-9]\.\d*\s+[A-Z]', r'^Chapter\s+\d+', r'^[A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+$'],
            'H3': [r'^[1-9]\.\d+\.\d+\s+[A-Z]', r'^[A-Z][a-z]+:', r'^Figure\s+\d+:', r'^Table\s+\d+:']
        }

        if re.match(r'^Eidesstattliche Versicherung$', text):
            return "H1"
        if re.match(r'^Dissertation$', text, re.IGNORECASE):
            return None

        for level, regex_list in patterns.items():
            for pattern in regex_list:
                if re.match(pattern, text, re.IGNORECASE):
                    return level

        font_size = block[3] - block[1]
        if font_size > 14 and 5 <= len(text.split()) <= 8:
            return "H2"

        return None

    def _is_signature_or_date(self, text: str) -> bool:
        patterns = [
            r'^\w+\,\s+den\s+\d{1,2}\.\s+\w+\s+\d{4}$',
            r'^\w+\,\s+\w+\s+\d{1,2}\,\s+\d{4}$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^Unterschrift$', r'^Signature$'
        ]
        return any(re.match(p, text) for p in patterns)

    def _is_document_metadata(self, text: str) -> bool:
        patterns = [
            r'^an der Fakultät', r'^der \w+ Universität', r'^vom \d{2}\.\d{2}\.\d{2}',
            r'^Promotionsordnung', r'^Dissertation$'
        ]
        return any(re.match(p, text, re.IGNORECASE) for p in patterns)

    def _clean_outline(self) -> List[Dict]:
        cleaned = []
        seen_texts = set()
        for item in sorted(self.outline, key=lambda x: (x['page'], x.get('type', 'z'))):
            norm_text = re.sub(r'[^a-zA-Z0-9]', '', item['text'].lower())
            if norm_text not in seen_texts and len(item['text']) > 3 and not item['text'].isdigit():
                cleaned.append(item)
                seen_texts.add(norm_text)
        return cleaned


def process_folder(input_dir: str, output_dir: str, use_ml=False):
    os.makedirs(output_dir, exist_ok=True)
    total_start = time.time()
    processed_files = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdf'):
            continue

        print(f"\nProcessing {filename}...")
        pdf_path = os.path.join(input_dir, filename)
        converter = EnhancedPDFToOutline(use_ml=use_ml)
        result = converter.convert_pdf(pdf_path)

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "title": result["title"],
                "outline": result["outline"]
            }, f, indent=2, ensure_ascii=False)

        processed_files += 1

    total_time = time.time() - total_start
    print(f"\nProcessed {processed_files} files in {total_time:.2f} seconds")
    print(f"Average time per file: {total_time/max(1, processed_files):.2f} seconds")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    process_folder(input_folder, output_folder, use_ml=True)
