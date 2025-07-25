import fitz  # PyMuPDF
import re
import json
import os
import tempfile
from typing import List, Dict, Optional, Tuple
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
os.environ["IOPATH_CACHE_DIR"] = os.path.join(os.getcwd(), "iopath_cache")
import layoutparser as lp
import torch

class EnhancedPDFToOutline:
    def __init__(self, use_ml=False, use_ocr=True):
        self.min_confidence = 0.5
        self.title = None
        self.outline = []
        self.current_page = 0
        self.use_ml = use_ml
        self.use_ocr = use_ocr
        self.ml_model = None
        self.layout_model = None
        self._initialize_ml_model()
        self._initialize_layout_model()

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
            
    def _initialize_layout_model(self):
        """Initialize LayoutParser model for document layout analysis"""
        try:
            self.layout_model = lp.Detectron2LayoutModel(
            config_path="C:/Users/Vivek Vasani/OneDrive/Desktop/AyeDobi/PubLayNet_model/config.yml",
            model_path="C:/Users/Vivek Vasani/OneDrive/Desktop/AyeDobi/PubLayNet_model/model_final.pth",
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.min_confidence],
            device="cuda" if torch.cuda.is_available() else "cpu"
)
        except Exception as e:
            print(f"Layout model initialization failed: {str(e)}")
            self.layout_model = None

    def _classify_pdf_type(self, doc: fitz.Document) -> str:
        """Enhanced PDF type classification with better scanned PDF detection"""
        text_pages = 0
        image_pages = 0
        total_text_length = 0
        total_images = 0
        
        for page_num in range(min(5, len(doc))):  # Check first 5 pages
            page = doc[page_num]
            text = page.get_text().strip()
            images = page.get_images()
            
            # Count text characteristics
            if text:
                text_pages += 1
                total_text_length += len(text)
                # Check if text seems like OCR artifacts (lots of single characters, weird spacing)
                if self._is_likely_ocr_artifact(text):
                    image_pages += 1
            
            # Count images that cover significant page area
            if images:
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        
                        # Get image dimensions
                        pix = fitz.Pixmap(image_data)
                        img_area = pix.width * pix.height
                        page_area = page.rect.width * page.rect.height
                        
                        # If image covers >70% of page, likely scanned
                        if img_area > 0.7 * page_area:
                            image_pages += 1
                            total_images += 1
                        pix = None
                    except:
                        continue
        
        # Enhanced classification logic
        pages_checked = min(5, len(doc))
        
        # If most pages have large images and little meaningful text
        if image_pages >= pages_checked * 0.6:
            return "Scanned PDF"
        
        # If average text per page is very low, likely scanned
        avg_text_per_page = total_text_length / max(1, pages_checked)
        if avg_text_per_page < 100 and total_images > 0:
            return "Scanned PDF"
        
        # If no extractable text at all
        if text_pages == 0:
            return "Scanned PDF"
        
        # If all pages have substantial text
        if text_pages == len(doc) and avg_text_per_page > 500:
            return "Native PDF"
        
        return "Mixed PDF"

    def _is_likely_ocr_artifact(self, text: str) -> bool:
        """Detect if text seems like poor OCR output"""
        if len(text) < 50:
            return True
        
        # Count single character "words"
        words = text.split()
        single_chars = sum(1 for word in words if len(word) == 1)
        
        # If >30% are single characters, likely OCR artifact
        if len(words) > 0 and single_chars / len(words) > 0.3:
            return True
        
        # Check for excessive spacing or weird characters
        weird_chars = sum(1 for char in text if not char.isalnum() and char not in ' .,!?-\n\t')
        if len(text) > 0 and weird_chars / len(text) > 0.2:
            return True
        
        return False

    def _extract_title(self, doc: fitz.Document):
        """Enhanced title extraction with better heuristics"""
        if len(doc) == 0:
            return
            
        # Try multiple approaches to find the title
        title_candidates = []
        
        # Method 1: Check PDF metadata
        metadata_title = doc.metadata.get('title', '').strip()
        if metadata_title and len(metadata_title) < 200:
            title_candidates.append((100, metadata_title, "metadata"))
        
        # Method 2: Analyze first page layout
        first_page = doc[0]
        text_dict = first_page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text or len(text) < 5:
                            continue
                        
                        # Skip obvious non-titles
                        if self._is_not_title(text):
                            continue
                        
                        # Calculate title score based on multiple factors
                        score = self._calculate_title_score(text, span, first_page)
                        if score > 30:
                            title_candidates.append((score, text, "layout"))
        
        # Method 3: Look for text blocks with large fonts at top of page
        blocks = first_page.get_text("blocks")
        page_height = first_page.rect.height
        
        for block in blocks:
            if len(block) < 5:
                continue
                
            text = block[4].strip()
            if not text or len(text) < 5:
                continue
            
            # Skip non-title patterns
            if self._is_not_title(text):
                continue
            
            # Position-based scoring (top 30% of page)
            y_pos = block[1]
            if y_pos < page_height * 0.3:
                # Font size estimation
                font_height = block[3] - block[1]
                # Centering check
                x_center = (block[0] + block[2]) / 2
                page_center = first_page.rect.width / 2
                centering_score = max(0, 50 - abs(x_center - page_center))
                
                score = font_height * 2 + centering_score
                if len(text.split()) >= 3 and len(text.split()) <= 15:
                    score += 20
                
                title_candidates.append((score, text, "block"))
        
        # Select best title candidate
        if title_candidates:
            title_candidates.sort(reverse=True, key=lambda x: x[0])
            best_title = title_candidates[0][1]
            
            # Clean up the title
            self.title = self._clean_title(best_title)

    def _is_not_title(self, text: str) -> bool:
        """Check if text is clearly not a title"""
        # Skip author names, dates, universities, etc.
        skip_patterns = [
            r'^(von|aus)\s+\w+',  # German: "von Name", "aus City"
            r'^\w+\s+\w+\s+\w+$',  # Three word names
            r'^\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}',  # Dates
            r'^(an der|der\s+\w+)\s+',  # "an der Universität", "der ... Universität"
            r'Universität|University|Institut|Institute',
            r'^Dissertation$',
            r'^(zur|zur Erlangung)',
            r'^\w+,\s+(den\s+)?\d',  # "Berlin, den 15."
            r'^Promotionsordnung',
            r'^(Prof\.|Dr\.|PhD)',
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    def _calculate_title_score(self, text: str, span: dict, page: fitz.Page) -> float:
        """Calculate title likelihood score"""
        score = 0
        
        # Font size score
        font_size = span.get("size", 0)
        score += font_size * 2
        
        # Bold text bonus
        if "bold" in span.get("font", "").lower():
            score += 20
        
        # Length penalty for very long text
        word_count = len(text.split())
        if 3 <= word_count <= 12:
            score += 15
        elif word_count > 20:
            score -= 30
        
        # Position bonus (top of page)
        bbox = span.get("bbox", [0, 0, 0, 0])
        if bbox[1] < page.rect.height * 0.25:
            score += 25
        
        # Centering bonus
        text_center = (bbox[0] + bbox[2]) / 2
        page_center = page.rect.width / 2
        if abs(text_center - page_center) < page.rect.width * 0.1:
            score += 15
        
        # Title-like patterns
        if re.search(r'\b(for|and|in|of|on|with|using|improving|enhanced)\b', text, re.IGNORECASE):
            score += 10
        
        return score

    def _clean_title(self, title: str) -> str:
        """Clean and format the extracted title"""
        # Remove excessive whitespace and newlines
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Remove trailing punctuation except necessary ones
        title = re.sub(r'[,;:\-_]+$', '', title)
        
        # Remove leading numbers or bullets
        title = re.sub(r'^[\d\.\-\•\*\s]+', '', title)
        
        return title.strip()

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    def _determine_psm(self, image: np.ndarray) -> int:
        """Determine optimal page segmentation mode"""
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Analyze text density
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Count non-white pixels
        text_pixels = np.sum(gray < 200)
        total_pixels = width * height
        text_density = text_pixels / total_pixels
        
        if text_density < 0.05:  # Very sparse text
            return 8  # Single word
        elif text_density < 0.15:  # Sparse text (titles, headings)
            return 7  # Single text line
        elif aspect_ratio > 2:  # Wide layout (multi-column)
            return 6  # Uniform block of text
        else:
            return 3  # Fully automatic page segmentation

    def _ocr_page(self, page_image: np.ndarray) -> str:
        """Enhanced OCR with multiple attempts and configurations"""
        processed = self._preprocess_image(page_image)
        psm = self._determine_psm(processed)
        
        # Try multiple OCR configurations
        configs = [
            f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()- ',
            f'--psm {psm} --oem 3',
            '--psm 6 --oem 3',
            '--psm 3 --oem 3'
        ]
        
        best_result = ""
        best_confidence = 0
        
        for config in configs:
            try:
                # Get OCR result with confidence scores
                data = pytesseract.image_to_data(
                    processed, config=config, output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text = ' '.join([data['text'][i] for i in range(len(data['text'])) 
                                   if int(data['conf'][i]) > 30])
                    
                    if avg_confidence > best_confidence and len(text.strip()) > len(best_result.strip()):
                        best_confidence = avg_confidence
                        best_result = text
                        
            except Exception as e:
                continue
        
        return best_result.strip()
    
    def _process_native_page(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Process a page with native text"""
        blocks = page.get_text("blocks")
        results = []
        
        for block in blocks:
            text = block[4].strip()
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

    def _process_scanned_page(self, page_image: np.ndarray, page_num: int) -> List[Dict]:
        """Enhanced scanned page processing"""
        results = []
        
        if self.layout_model:
            # Use layout analysis if available
            try:
                layout = self.layout_model.detect(page_image)
                for block in layout:
                    x1, y1, x2, y2 = map(int, [block.block.x_1, block.block.y_1, 
                                             block.block.x_2, block.block.y_2])
                    
                    # Add padding to avoid cutting text
                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding) 
                    x2 = min(page_image.shape[1], x2 + padding)
                    y2 = min(page_image.shape[0], y2 + padding)
                    
                    cropped = page_image[y1:y2, x1:x2]
                    if cropped.size > 0:
                        text = self._ocr_page(cropped)
                        
                        if text and len(text.strip()) > 2:
                            level = self._determine_heading_level_from_layout(text, block)
                            if level:
                                results.append({
                                    "level": level,
                                    "text": text,
                                    "page": page_num,
                                    "type": block.type.lower()
                                })
            except Exception as e:
                print(f"Layout processing failed for page {page_num}: {str(e)}")
        
        # Fallback: segment page and OCR different regions
        if not results:
            results = self._segment_and_ocr_page(page_image, page_num)
                
        return results

    def _determine_heading_level_from_layout(self, text: str, layout_block) -> Optional[str]:
        """Determine heading level using layout information"""
        if len(text) > 200 or len(text.split()) > 20:
            return None
        
        block_type = layout_block.type.lower()
        block_height = layout_block.block.y_2 - layout_block.block.y_1
        
        # Use layout type first
        if block_type == "title":
            return "H1"
        
        # Use size and text patterns
        patterns = {
            'H1': [r'^[A-Z][A-Za-z\s]{10,}$', r'^(Abstract|Introduction|Conclusion|References?)\b'],
            'H2': [r'^[1-9]\.\s+[A-Z]', r'^Chapter\s+\d+', r'^\d+\.\d+\s+[A-Z]'],
            'H3': [r'^\d+\.\d+\.\d+\s+[A-Z]', r'^[A-Z][a-z]+:', r'^(Figure|Table)\s+\d+']
        }
        
        for level, regex_list in patterns.items():
            for pattern in regex_list:
                if re.match(pattern, text, re.IGNORECASE):
                    return level
        
        # Size-based fallback
        if block_height > 25:
            return "H1"
        elif block_height > 15:
            return "H2"
        elif len(text.split()) <= 10:
            return "H3"
        
        return None

    def _segment_and_ocr_page(self, page_image: np.ndarray, page_num: int) -> List[Dict]:
        """Segment page into regions and OCR each"""
        results = []
        height, width = page_image.shape[:2]
        
        # Simple horizontal segmentation
        segment_height = height // 10  # Divide into 10 horizontal segments
        
        for i in range(0, height, segment_height):
            y1 = i
            y2 = min(i + segment_height + 20, height)  # Overlap to avoid cutting text
            
            segment = page_image[y1:y2, 0:width]
            text = self._ocr_page(segment)
            
            if text and len(text.strip()) > 3:
                # Simple heuristic: text at top is more likely to be heading
                position_score = (height - i) / height
                level = "H1" if position_score > 0.8 else "H2" if position_score > 0.6 else "H3"
                
                # Refine based on text characteristics
                refined_level = self._determine_heading_level(text, None)
                if refined_level:
                    level = refined_level
                
                results.append({
                    "level": level,
                    "text": text,
                    "page": page_num,
                    "type": "text"
                })
        
        return results

    def convert_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Convert PDF to structured outline with enhanced processing"""
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            pdf_type = self._classify_pdf_type(doc)
            print(f"PDF Type detected: {pdf_type}")
            
            self._extract_title(doc)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    if pdf_type == "Native PDF" or (pdf_type == "Mixed PDF" and len(page.get_text().strip()) > 100):
                        # Process as native PDF
                        futures.append(executor.submit(
                            self._process_native_page, page, page_num + 1
                        ))
                    elif self.use_ocr and pdf_type in ["Scanned PDF", "Mixed PDF"]:
                        # Convert page to image for OCR processing
                        try:
                            pix = page.get_pixmap(dpi=200)
                            img_data = pix.samples
                            img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                                pix.height, pix.width, pix.n
                            )
                            
                            # Convert RGBA to RGB if necessary
                            if img.shape[2] == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                            elif img.shape[2] == 1:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                                
                            futures.append(executor.submit(
                                self._process_scanned_page, img, page_num + 1
                            ))
                            pix = None  # Free memory
                        except Exception as e:
                            print(f"Failed to process page {page_num + 1} as image: {str(e)}")
                            continue
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        self.outline.extend(result)
                    except Exception as e:
                        print(f"Error processing page: {str(e)}")
                        continue
                        
            doc.close()
            
            return {
                "title": self.title or os.path.basename(pdf_path),
                "outline": self._clean_outline(),
                "pdf_type": pdf_type,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "title": os.path.basename(pdf_path),
                "outline": [],
                "pdf_type": "Error",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

    def _process_page(self, page, page_num):
        """Process a single page using layout analysis"""
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
        """Extract figures and tables from mixed PDFs using OCR"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                images = convert_from_path(pdf_path, output_folder=tmpdir, fmt='jpeg', first_page=1, last_page=5)
                for i, img in enumerate(images, start=1):
                    ocr_text = pytesseract.image_to_string(np.array(img))
                    self._process_ocr_text(ocr_text, i)
        except Exception as e:
            print(f"OCR processing skipped: {str(e)}")

    def _process_ocr_text(self, text: str, page_num: int):
        """Process OCR text to extract figures and tables"""
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
        """Clean and normalize text"""
        patterns = [
            r'\.{3,}\s*\d*$', r'^\d+\s*[-.]\s*', r'\s*\d+\s*$', r'[\x00-\x1F\x7F-\x9F]'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return ' '.join(text.split()).strip()

    def _determine_heading_level(self, text: str, block=None) -> Optional[str]:
        """Determine heading level using heuristic rules"""
        if len(text) > 200 or len(text.split()) > 15:
            return None
            
        # Check common heading patterns
        patterns = {
            'H1': [r'^[IVX]+\.', r'^Chapter \d+', r'^[A-Z][A-Z\s]{10,}$'],
            'H2': [r'^\d+\.\d+', r'^[A-Z][a-z]+:', r'^[A-Z][a-z]+\s[A-Z][a-z]+$'],
            'H3': [r'^\d+\.\d+\.\d+', r'^[a-z]\)', r'^•\s']
        }
        
        for level, regex_list in patterns.items():
            if any(re.search(pattern, text) for pattern in regex_list):
                return level
                
        # Fallback to font size analysis if available
        if block and len(block) >= 4:
            font_size = block[3] - block[1]
            if font_size > 14:
                return "H2"
                
        return None

    def _determine_heading_level_rules(self, text: str, block=None) -> Optional[str]:
        """Determine heading level using heuristic rules"""
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

        if block and hasattr(block, 'height'):
            font_size = block.height
            if font_size > 14 and 5 <= len(text.split()) <= 8:
                return "H2"
        elif block and len(block) >= 4:  # For fitz blocks
            font_size = block[3] - block[1]
            if font_size > 14 and 5 <= len(text.split()) <= 8:
                return "H2"
        elif 5 <= len(text.split()) <= 8:
            return "H2"

        return None

    def _is_signature_or_date(self, text: str) -> bool:
        """Check if text is a signature or date"""
        patterns = [
            r'^\w+\,\s+den\s+\d{1,2}\.\s+\w+\s+\d{4}$',
            r'^\w+\,\s+\w+\s+\d{1,2}\,\s+\d{4}$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^Unterschrift$', r'^Signature$'
        ]
        return any(re.match(p, text) for p in patterns)

    def _is_document_metadata(self, text: str) -> bool:
        """Check if text is document metadata"""
        patterns = [
            r'^an der Fakultät', r'^der \w+ Universität', r'^vom \d{2}\.\d{2}\.\d{2}',
            r'^Promotionsordnung', r'^Dissertation$'
        ]
        return any(re.match(p, text, re.IGNORECASE) for p in patterns)

    def _clean_outline(self) -> List[Dict]:
        """Clean and deduplicate outline entries"""
        cleaned = []
        seen_texts = set()
        
        for item in sorted(self.outline, key=lambda x: x['page']):
            norm_text = re.sub(r'\W+', '', item['text'].lower())
            if norm_text not in seen_texts and len(item['text']) > 3:
                cleaned.append(item)
                seen_texts.add(norm_text)
                
        return cleaned

def process_pdfs(input_dir: str, output_dir: str):
    """Batch process all PDFs in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    total_start = time.time()
    processed_files = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdf'):
            continue
            
        print(f"Processing {filename}...")
        converter = EnhancedPDFToOutline()
        result = converter.convert_pdf(os.path.join(input_dir, filename))
        
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "title": result["title"],
                "outline": result["outline"],
                "pdf_type": result.get("pdf_type", "unknown"),
                "error": result.get("error", None)
            }, f, indent=2, ensure_ascii=False)

        processed_files += 1

    total_time = time.time() - total_start
    print(f"\nProcessed {processed_files} files in {total_time:.2f} seconds")
    print(f"Average time per file: {total_time/max(1, processed_files):.2f} seconds")


if __name__ == "__main__":
    # Configure Tesseract path if needed
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    
    input_folder = "input"
    output_folder = "output"
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Enable ML for heading detection and OCR for scanned PDFs
    process_pdfs(input_folder, output_folder)