from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

class PDFLanguageDetector:
    def __init__(self):
        """Initialize langdetect with reproducible results"""
        DetectorFactory.seed = 42  # For consistent results
        # Newer versions of langdetect don't have load_all_profiles()
        # So we'll use the default behavior
        self.min_text_length = 15

    def detect_language(self, text: str) -> str:
        """
        Pure statistical language detection using langdetect
        No rules or whitelists - just what langdetect returns
        """
        if not text or len(text.strip()) < self.min_text_length:
            return "unknown"
        
        try:
            # Let langdetect handle everything statistically
            return detect(text)
        except LangDetectException:
            return "unknown"
        except Exception as e:
            print(f"Language detection error: {str(e)}")
            return "unknown"