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

class EnhancedPDFToOutline:
    def __init__(self, use_ml=False, use_ocr=True):
        self.min_confidence = 0.5
        self.title = None
        self.outline = []
        self.current_page = 0
        self.use_ml = use_ml
        self.use_ocr = use_ocr
        self.min_text_length = 15
        
        # Initialize components
        self.language_detector = PDFLanguageDetector()
        self.ml_models = PDFMLModels(use_ml=use_ml)
        self.ocr_processor = PDFOCRProcessor()
        self.pdf_utils = PDFProcessingUtils()
        self.utils = PDFUtils()
        
        self.ml_model = self.ml_models.ml_model
        self.layout_model = self.ml_models.layout_model

    def convert_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Convert PDF to structured outline with enhanced processing"""
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            pdf_type = self.pdf_utils.classify_pdf_type(doc)
            print(f"PDF Type detected: {pdf_type}")
            
            self.pdf_utils.extract_title(doc, self)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    
                    if pdf_type == "Native PDF" or (pdf_type == "Mixed PDF" and len(page.get_text().strip()) > 100):
                        # Process as native PDF
                        futures.append(executor.submit(
                            self.pdf_utils.process_native_page, self, page, page_num + 1
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
                                self.ocr_processor.process_scanned_page, self, img, page_num + 1
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