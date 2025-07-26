import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Optional HuggingFace model (mBERT or TinyLlama multilingual variant)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PDFLanguageDetector:
    def __init__(self, min_text_length: int = 15):
        """
        Hybrid language detector using langdetect and LLM fallback (e.g., TinyLlama or mBERT).
        """
        DetectorFactory.seed = 42  # Ensure reproducible langdetect results
        self.min_text_length = min_text_length
        self.model_name = "papluca/xlm-roberta-base-language-detection"  # multilingual fallback
        self._load_llm_model()

    def _load_llm_model(self):
        try:
            print("Loading fallback multilingual LLM model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.label_map = self.model.config.id2label
        except Exception as e:
            print(f"[Warning] LLM model loading failed: {e}")
            self.tokenizer = None
            self.model = None
            self.label_map = {}

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning before detection (optional for noisy OCR text)
        """
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        return text.strip()

    def detect_language_statistical(self, text: str) -> str:
        """
        Fast detection using langdetect (statistical, non-neural)
        """
        text = self.clean_text(text)
        if not text or len(text) < self.min_text_length:
            return "unknown"

        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
        except Exception as e:
            print(f"[LangDetect Error] {e}")
            return "unknown"

    def detect_language_llm(self, text: str) -> str:
        """
        Fallback LLM-based language detection (TinyLlama, mBERT or XLM-Roberta)
        """
        if self.model is None or self.tokenizer is None:
            return "unknown"

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted = torch.argmax(logits, dim=1)
            return self.label_map.get(predicted.item(), "unknown")
        except Exception as e:
            print(f"[LLM Detection Error] {e}")
            return "unknown"

    def detect_language(self, text: str) -> str:
        """
        Hybrid detection: try langdetect first, then fallback to LLM
        """
        lang = self.detect_language_statistical(text)

        # fallback if uncertain or too long for langdetect
        if lang == "unknown" or len(text) > 1000:
            lang_llm = self.detect_language_llm(text)
            return lang_llm or lang
        return lang
