import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import time

class PDFLanguageDetector:
    def __init__(self, min_text_length: int = 15, use_quantized: bool = True):
        """
        Optimized hybrid language detector with batch processing
        """
        DetectorFactory.seed = 42
        self.min_text_length = min_text_length
        self.model_name = "papluca/xlm-roberta-base-language-detection"
        self.translation_model_name = "Helsinki-NLP/opus-mt-mul-en"
        
        # Performance tracking
        self.batch_size = 32
        self.max_workers = 4
        self.lang_cache = defaultdict(str)
        self.en_keywords = {'the', 'and', 'for', 'are', 'this', 'that'}
        
        self._load_models(use_quantized)
        
    def _load_models(self, use_quantized: bool):
        """Load optimized models with batch support"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch_dtype = torch.float16 if use_quantized and device == 'cuda' else torch.float32
            
            # Language detection model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype
            ).to(device).eval()
            self.label_map = self.model.config.id2label
            
            # Translation pipeline with batch support
            self.translator = pipeline(
                "translation",
                model=self.translation_model_name,
                device=0 if device == 'cuda' else -1,
                torch_dtype=torch_dtype,
                batch_size=self.batch_size
            )
            
            print(f"Models loaded on {device.upper()} with batch size {self.batch_size}")
        except Exception as e:
            print(f"[Warning] Model loading failed: {e}")
            self.tokenizer = None
            self.model = None
            self.translator = None
            self.label_map = {}

    def clean_text(self, text: str) -> str:
        """Fast text cleaning with regex compilation"""
        if not hasattr(self, '_whitespace_re'):
            self._whitespace_re = re.compile(r'\s+')
            self._unicode_re = re.compile(r'[\u200b-\u200f\u202a-\u202e]')
        
        text = self._whitespace_re.sub(' ', text)
        text = self._unicode_re.sub('', text)
        return text.strip()

    def _is_likely_english(self, text: str) -> bool:
        """Fast English detection for common cases"""
        words = set(text.lower().split())
        return len(words & self.en_keywords) >= 2

    def detect_language_statistical(self, text: str) -> str:
        """Optimized statistical detection with caching"""
        if len(text) < self.min_text_length:
            return "unknown"
            
        # Check cache first
        cache_key = hash(text[:100])  # First 100 chars as key
        if cache_key in self.lang_cache:
            return self.lang_cache[cache_key]
        
        try:
            lang = detect(text)
            self.lang_cache[cache_key] = lang
            return lang
        except (LangDetectException, Exception):
            return "unknown"

    def _batch_detect_llm(self, texts: List[str]) -> List[str]:
        """Batch process texts with LLM model"""
        if not self.model or not self.tokenizer:
            return ["unknown"] * len(texts)
            
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128  # Reduced for speed
            ).to(self.model.device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = self.model(**inputs).logits
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
            
            return [self.label_map.get(p, "unknown") for p in predicted]
        except Exception as e:
            print(f"[Batch LLM Error] {e}")
            return ["unknown"] * len(texts)

    def _batch_translate(self, texts: List[str], src_langs: List[str]) -> List[str]:
        """Batch translate non-English texts"""
        if not self.translator:
            return texts
            
        try:
            # Group by language for more efficient translation
            lang_groups = defaultdict(list)
            for i, (text, lang) in enumerate(zip(texts, src_langs)):
                if lang != 'en':
                    lang_groups[lang].append((i, text))
            
            # Translate each language group
            translated_texts = texts.copy()
            for lang, items in lang_groups.items():
                indices, lang_texts = zip(*items)
                results = self.translator(lang_texts, src_lang=lang)
                for idx, result in zip(indices, results):
                    translated_texts[idx] = result['translation_text']
            
            return translated_texts
        except Exception as e:
            print(f"[Batch Translate Error] {e}")
            return texts

    def detect_language(self, text: str) -> str:
        """Single text language detection interface"""
        if not text:
            return "unknown"
            
        text = self.clean_text(text)
        
        # First try fast statistical method
        lang = self.detect_language_statistical(text)
        
        # Fallback to LLM if needed
        if lang == "unknown" or len(text) > 1000:
            lang_llm = self._batch_detect_llm([text])[0]
            return lang_llm if lang_llm != "unknown" else lang
            
        return lang

    def detect_and_translate(self, text: str) -> Tuple[str, str]:
        """Detect language and translate if not English"""
        lang = self.detect_language(text)
        if lang != 'en':
            translated = self._batch_translate([text], [lang])[0]
            return lang, translated
        return lang, text

    def process_batch(self, texts: List[str]) -> List[Tuple[str, str]]:
        """Optimized batch processing pipeline"""
        if not texts:
            return []
            
        # Clean all texts first
        cleaned_texts = [self.clean_text(t) for t in texts]
        
        # Step 1: Fast statistical detection in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            stats_langs = list(executor.map(self.detect_language_statistical, cleaned_texts))
        
        # Step 2: Identify texts needing LLM detection
        llm_indices = [i for i, lang in enumerate(stats_langs) 
                      if lang == "unknown" or len(cleaned_texts[i]) > 1000]
        llm_texts = [cleaned_texts[i] for i in llm_indices]
        
        # Batch process LLM texts
        if llm_texts:
            llm_results = self._batch_detect_llm(llm_texts)
            for idx, lang in zip(llm_indices, llm_results):
                stats_langs[idx] = lang
        
        # Step 3: Batch translation for non-English texts
        translated_texts = self._batch_translate(cleaned_texts, stats_langs)
        
        return list(zip(stats_langs, translated_texts))