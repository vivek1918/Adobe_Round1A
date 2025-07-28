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
os.environ["IOPATH_CACHE_DIR"] = os.path.join(os.getcwd(), "iopath_cache")
import layoutparser as lp
import torch
from language_detection import PDFLanguageDetector
from ml_models import PDFMLModels
from ocr_processing import PDFOCRProcessor
from pdf_processing import PDFProcessingUtils
from utils import PDFUtils
from functools import partial
import os
import torch
if not torch.cuda.is_available():
    torch.set_default_device("cpu")  # Silences "CUDA not available" warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables tokenizer fork warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

class EnhancedPDFToOutline:
    def __init__(self, use_ml=False, use_ocr=True, language_detector=None, ocr_processor=None):
        self.min_confidence = 0.5
        self.title = None
        self.outline = []
        self.current_page = 0
        self.use_ml = use_ml
        self.use_ocr = use_ocr
        self.min_text_length = 15
        
        # Initialize components
        self.language_detector = language_detector if language_detector else PDFLanguageDetector()
        self.ml_models = PDFMLModels(use_ml=use_ml)
        self.ocr_processor = ocr_processor if ocr_processor else PDFOCRProcessor()
        self.pdf_utils = PDFProcessingUtils()
        self.utils = PDFUtils()
        
        self.ml_model = self.ml_models.ml_model
        self.layout_model = self.ml_models.layout_model
    
    @staticmethod
    def _process_page_worker(pdf_path: str, job_info, detector):
        task_type = job_info[0]
        page_num = job_info[1]

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            text = ""

            if task_type == "native":
                text = page.get_text().strip()
            elif task_type == "ocr":
                img = job_info[2]
                if img is not None:
                    text = pytesseract.image_to_string(img)

            if len(text.strip()) == 0:
                return []

            # Detect language using passed-in detector
            language = detector.detect_language(text)

            return [{
                "level": "H2",  # placeholder for now
                "text": text.strip(),
                "page": page_num + 1,
                "type": "text",
                "language": language
            }]

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            return []

    def convert_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Convert PDF to structured outline with enhanced processing and multiprocessing"""
        import multiprocessing
        start_time = time.time()

        try:
            doc = fitz.open(pdf_path)
            pdf_type = self.pdf_utils.classify_pdf_type(doc)
            print(f"PDF Type detected: {pdf_type}")
            self.pdf_utils.extract_title(doc, self)

            page_jobs = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                if pdf_type == "Native PDF" or (pdf_type == "Mixed PDF" and len(text) > 100):
                    page_jobs.append(("native", page_num))
                elif self.use_ocr and pdf_type in ["Scanned PDF", "Mixed PDF"]:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img_data = pix.samples
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                            pix.height, pix.width, pix.n
                        )
                        if img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        elif img.shape[2] == 1:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        page_jobs.append(("ocr", page_num, img))
                        pix = None
                    except Exception as e:
                        print(f"Failed to process page {page_num + 1}: {str(e)}")

            worker_func = partial(
                EnhancedPDFToOutline._process_page_worker,
                pdf_path,
                detector=self.language_detector
            )

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(worker_func, job) for job in page_jobs]
                results = [f.result() for f in futures]

            for result in results:
                if result:
                    self.outline.extend(result)

            doc.close()

            return {
                "title": self.title or os.path.basename(pdf_path),
                "outline": self.utils.clean_outline(self.outline),
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