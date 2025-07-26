import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Optional, Tuple
import re
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from language_detection import PDFLanguageDetector

class PDFOCRProcessor:
    def __init__(self, language_detector: Optional[PDFLanguageDetector] = None):
        self.min_confidence = 0.5
        self.language_detector = language_detector
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.tess_lang_map = {
            'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu',
            'it': 'ita', 'pt': 'por', 'ru': 'rus', 'zh': 'chi_sim',
            'ja': 'jpn', 'ar': 'ara', 'hi': 'hin'
        }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Optimized image preprocessing pipeline"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _get_tesseract_config(self, lang_code: str = 'en') -> str:
        """Get optimized Tesseract config for language"""
        tess_lang = self.tess_lang_map.get(lang_code, 'eng')
        return f'--psm 6 --oem 3 -l {tess_lang}'

    def _ocr_with_lang(self, image: np.ndarray, lang: str) -> Tuple[str, float]:
        """Perform OCR with language-specific optimization"""
        config = self._get_tesseract_config(lang)
        try:
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            if not confidences:
                return "", 0.0
                
            avg_conf = sum(confidences) / len(confidences)
            text = ' '.join([
                data['text'][i] for i in range(len(data['text']))
                if int(data['conf'][i]) > 30
            ]).strip()
            
            return text, avg_conf
        except Exception as e:
            print(f"OCR Error for lang {lang}: {e}")
            return "", 0.0

    def _ocr_page(self, page_image: np.ndarray, detected_lang: str = 'en') -> str:
        """Multilingual OCR with fallback"""
        processed = self._preprocess_image(page_image)
        primary_text, primary_conf = self._ocr_with_lang(processed, detected_lang)
        
        # Fallback to English if low confidence
        if primary_conf < 40 and detected_lang != 'en':
            eng_text, eng_conf = self._ocr_with_lang(processed, 'en')
            if eng_conf > primary_conf + 10:
                return eng_text
                
        return primary_text

    def process_scanned_page(self, converter, page_image: np.ndarray, page_num: int) -> List[Dict]:
        """Process page with multilingual support"""
        results = []
        if converter.layout_model:
            try:
                layout = converter.layout_model.detect(page_image)
                for block in layout:
                    x1, y1, x2, y2 = map(int, [block.block.x_1, block.block.y_1, 
                                               block.block.x_2, block.block.y_2])
                    cropped = page_image[y1:y2, x1:x2]
                    if cropped.size > 0:
                        # Initial OCR in detected language
                        text = self._ocr_page(cropped)
                        if len(text.strip()) > 2:
                            # Detect language and translate if needed
                            lang, translated = converter.language_detector.detect_and_translate(text)
                            level = converter.pdf_utils.determine_heading_level_from_layout(text, block)
                            
                            results.append({
                                "level": level or "H3",
                                "text": text,
                                "translated": translated if lang != 'en' else None,
                                "page": page_num,
                                "type": block.type.lower(),
                                "language": lang
                            })
            except Exception as e:
                print(f"Layout processing failed: {e}")

        if not results:
            results = self._segment_and_ocr_page(converter, page_image, page_num)
        return results

    def _process_single_page(self, converter, page_tuple):
        """Wrapper for parallel processing"""
        page_num, page_image = page_tuple
        return self.process_scanned_page(converter, page_image, page_num)

    def process_pdf_pages_in_parallel(self, converter, pdf_images: List[np.ndarray]) -> List[Dict]:
        """Process pages with multithreading"""
        with self.executor as executor:
            process_func = partial(self._process_single_page, converter)
            page_tuples = list(enumerate(pdf_images, start=1))
            all_results = list(executor.map(process_func, page_tuples))
        return [item for sublist in all_results for item in sublist]