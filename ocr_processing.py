import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Optional
import re
import multiprocessing
from functools import partial

class PDFOCRProcessor:
    def __init__(self):
        self.min_confidence = 0.5

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _determine_psm(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        aspect_ratio = width / height
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        text_pixels = np.sum(gray < 200)
        total_pixels = width * height
        text_density = text_pixels / total_pixels

        if text_density < 0.05:
            return 8
        elif text_density < 0.15:
            return 7
        elif aspect_ratio > 2:
            return 6
        else:
            return 3

    def _ocr_page(self, page_image: np.ndarray) -> str:
        processed = self._preprocess_image(page_image)
        psm = self._determine_psm(processed)
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
                data = pytesseract.image_to_data(
                    processed, config=config, output_type=pytesseract.Output.DICT
                )
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text = ' '.join([
                        data['text'][i] for i in range(len(data['text']))
                        if int(data['conf'][i]) > 30
                    ])
                    if avg_confidence > best_confidence and len(text.strip()) > len(best_result.strip()):
                        best_confidence = avg_confidence
                        best_result = text
            except Exception:
                continue
        return best_result.strip()

    def process_scanned_page(self, converter, page_image: np.ndarray, page_num: int) -> List[Dict]:
        results = []
        if converter.layout_model:
            try:
                layout = converter.layout_model.detect(page_image)
                for block in layout:
                    x1, y1, x2, y2 = map(int, [block.block.x_1, block.block.y_1, 
                                               block.block.x_2, block.block.y_2])
                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding) 
                    x2 = min(page_image.shape[1], x2 + padding)
                    y2 = min(page_image.shape[0], y2 + padding)
                    cropped = page_image[y1:y2, x1:x2]
                    if cropped.size > 0:
                        text = self._ocr_page(cropped)
                        if text and len(text.strip()) > 2:
                            level = converter.pdf_utils.determine_heading_level_from_layout(text, block)
                            if level:
                                language = converter.language_detector.detect_language(text)
                                results.append({
                                    "level": level,
                                    "text": text,
                                    "page": page_num,
                                    "type": block.type.lower(),
                                    "language": language
                                })
            except Exception as e:
                print(f"Layout processing failed for page {page_num}: {str(e)}")

        if not results:
            results = self._segment_and_ocr_page(converter, page_image, page_num)
        return results

    def _segment_and_ocr_page(self, converter, page_image: np.ndarray, page_num: int) -> List[Dict]:
        results = []
        height, width = page_image.shape[:2]
        segment_height = height // 10
        for i in range(0, height, segment_height):
            y1 = i
            y2 = min(i + segment_height + 20, height)
            segment = page_image[y1:y2, 0:width]
            text = self._ocr_page(segment)
            if text and len(text.strip()) > 3:
                position_score = (height - i) / height
                level = "H1" if position_score > 0.8 else "H2" if position_score > 0.6 else "H3"
                refined_level = converter.pdf_utils.determine_heading_level(text, None)
                if refined_level:
                    level = refined_level
                language = converter.language_detector.detect_language(text)
                results.append({
                    "level": level,
                    "text": text,
                    "page": page_num,
                    "type": "text",
                    "language": language
                })
        return results

    def _process_single_page(self, converter, page_tuple):
        page_num, page_image = page_tuple
        return self.process_scanned_page(converter, page_image, page_num)

    def process_pdf_pages_in_parallel(self, converter, pdf_images: List[np.ndarray], num_workers: int = 4) -> List[Dict]:
        with multiprocessing.Pool(processes=num_workers) as pool:
            process_func = partial(self._process_single_page, converter)
            page_tuples = list(enumerate(pdf_images, start=1))
            all_results = pool.map(process_func, page_tuples)
        # Flatten the list of lists
        return [item for sublist in all_results for item in sublist]


