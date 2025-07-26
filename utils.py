import re
from typing import List, Dict

class PDFUtils:
    def clean_outline(self, outline: List[Dict]) -> List[Dict]:
        """Clean and deduplicate outline entries"""
        cleaned = []
        seen_texts = set()
        
        for item in sorted(outline, key=lambda x: x['page']):
            norm_text = re.sub(r'\W+', '', item['text'].lower())
            if norm_text not in seen_texts and len(item['text']) > 3:
                cleaned.append(item)
                seen_texts.add(norm_text)
        return cleaned

    def is_meaningful_text(self, text: str) -> bool:
        """Minimal validation to filter complete gibberish"""
        if not text or len(text) < 5:
            return False
        
        words = text.split()
        if not words:
            return False
            
        valid_words = sum(1 for word in words if 2 <= len(word) <= 25)
        return valid_words / len(words) > 0.5

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        patterns = [
            r'\.{3,}\s*\d*$', r'^\d+\s*[-.]\s*', r'\s*\d+\s*$', r'[\x00-\x1F\x7F-\x9F]'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return ' '.join(text.split()).strip()