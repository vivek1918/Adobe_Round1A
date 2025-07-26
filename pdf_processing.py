import fitz
import re
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

class PDFProcessingUtils:
    def __init__(self):
        self.heading_model = None
        try:
            self.heading_model = joblib.load('heading_classifier.joblib')
            self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
        except:
            self.heading_model = None
    def classify_pdf_type(self, doc: fitz.Document) -> str:
        """Enhanced PDF type classification with better scanned PDF detection"""
        text_pages = 0
        image_pages = 0
        total_text_length = 0
        total_images = 0
        
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            text = page.get_text().strip()
            images = page.get_images()
            
            if text:
                text_pages += 1
                total_text_length += len(text)
                if self._is_likely_ocr_artifact(text):
                    image_pages += 1
            
            if images:
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        pix = fitz.Pixmap(image_data)
                        img_area = pix.width * pix.height
                        page_area = page.rect.width * page.rect.height
                        if img_area > 0.7 * page_area:
                            image_pages += 1
                            total_images += 1
                        pix = None
                    except:
                        continue
        
        pages_checked = min(5, len(doc))
        if image_pages >= pages_checked * 0.6:
            return "Scanned PDF"
        
        avg_text_per_page = total_text_length / max(1, pages_checked)
        if avg_text_per_page < 100 and total_images > 0:
            return "Scanned PDF"
        
        if text_pages == 0:
            return "Scanned PDF"
        
        if text_pages == len(doc) and avg_text_per_page > 500:
            return "Native PDF"
        
        return "Mixed PDF"

    def _is_likely_ocr_artifact(self, text: str) -> bool:
        """Detect if text seems like poor OCR output"""
        if len(text) < 50:
            return True
        
        words = text.split()
        single_chars = sum(1 for word in words if len(word) == 1)
        if len(words) > 0 and single_chars / len(words) > 0.3:
            return True
        
        weird_chars = sum(1 for char in text if not char.isalnum() and char not in ' .,!?-\n\t')
        if len(text) > 0 and weird_chars / len(text) > 0.2:
            return True
        
        return False

    def extract_title(self, doc: fitz.Document, converter):
        """Enhanced title extraction with better heuristics"""
        if len(doc) == 0:
            return
            
        title_candidates = []
        metadata_title = doc.metadata.get('title', '').strip()
        if metadata_title and len(metadata_title) < 200:
            title_candidates.append((100, metadata_title, "metadata"))
        
        first_page = doc[0]
        text_dict = first_page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text or len(text) < 5:
                            continue
                        if self._is_not_title(text):
                            continue
                        score = self._calculate_title_score(text, span, first_page)
                        if score > 30:
                            title_candidates.append((score, text, "layout"))
        
        blocks = first_page.get_text("blocks")
        page_height = first_page.rect.height
        
        for block in blocks:
            if len(block) < 5:
                continue
            text = block[4].strip()
            if not text or len(text) < 5:
                continue
            if self._is_not_title(text):
                continue
            
            y_pos = block[1]
            if y_pos < page_height * 0.3:
                font_height = block[3] - block[1]
                x_center = (block[0] + block[2]) / 2
                page_center = first_page.rect.width / 2
                centering_score = max(0, 50 - abs(x_center - page_center))
                score = font_height * 2 + centering_score
                if len(text.split()) >= 3 and len(text.split()) <= 15:
                    score += 20
                title_candidates.append((score, text, "block"))
        
        if title_candidates:
            title_candidates.sort(reverse=True, key=lambda x: x[0])
            best_title = title_candidates[0][1]
            converter.title = self._clean_title(best_title)

    def _is_not_title(self, text: str) -> bool:
        """Check if text is clearly not a title"""
        skip_patterns = [
            r'^(von|aus)\s+\w+', r'^\w+\s+\w+\s+\w+$', r'^\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}',
            r'^(an der|der\s+\w+)\s+', r'Universität|University|Institut|Institute',
            r'^Dissertation$', r'^(zur|zur Erlangung)', r'^\w+,\s+(den\s+)?\d',
            r'^Promotionsordnung', r'^(Prof\.|Dr\.|PhD)',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in skip_patterns)

    def _calculate_title_score(self, text: str, span: dict, page: fitz.Page) -> float:
        """Calculate title likelihood score"""
        score = 0
        font_size = span.get("size", 0)
        score += font_size * 2
        
        if "bold" in span.get("font", "").lower():
            score += 20
        
        word_count = len(text.split())
        if 3 <= word_count <= 12:
            score += 15
        elif word_count > 20:
            score -= 30
        
        bbox = span.get("bbox", [0, 0, 0, 0])
        if bbox[1] < page.rect.height * 0.25:
            score += 25
        
        text_center = (bbox[0] + bbox[2]) / 2
        page_center = page.rect.width / 2
        if abs(text_center - page_center) < page.rect.width * 0.1:
            score += 15
        
        if re.search(r'\b(for|and|in|of|on|with|using|improving|enhanced)\b', text, re.IGNORECASE):
            score += 10
        
        return score

    def _clean_title(self, title: str) -> str:
        """Clean and format the extracted title"""
        title = re.sub(r'\s+', ' ', title.strip())
        title = re.sub(r'[,;:\-_]+$', '', title)
        title = re.sub(r'^[\d\.\-\•\*\s]+', '', title)
        return title.strip()

    def process_native_page(self, converter, page: fitz.Page, page_num: int) -> List[Dict]:
        """Process a page with native text"""
        blocks = page.get_text("blocks")
        results = []
        
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            level = self.determine_heading_level(text, block)
            if level:
                language = converter.language_detector.detect_language(text)
                results.append({
                    "level": level,
                    "text": text,
                    "page": page_num,
                    "type": "text",
                    "language": language
                })
        return results

    def determine_heading_level_ml(self, text: str, block=None) -> Optional[str]:
        """Machine learning based heading level detection"""
        if not self.heading_model or len(text) > 500:
            return self.determine_heading_level(text, block)  # Fallback to heuristic
            
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n') + 1,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text),
            'ends_with_punct': int(text.strip()[-1] in {':', ';'}),
            'has_number': int(any(c.isdigit() for c in text)),
            'font_size': block[3] - block[1] if block and len(block) >= 4 else 12
        }
        
        # Vectorize text
        text_features = self.vectorizer.transform([text])
        
        # Combine all features
        import numpy as np
        X = np.hstack([
            text_features.toarray(),
            np.array([[features[k] for k in sorted(features.keys())]])
        ])
        
        prediction = self.heading_model.predict(X)
        return f"H{prediction[0]}" if prediction[0] in {1, 2, 3} else None