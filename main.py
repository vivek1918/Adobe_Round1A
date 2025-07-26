import os
import json
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf_outline_converter import EnhancedPDFToOutline
from language_detection import PDFLanguageDetector
from ocr_processing import PDFOCRProcessor

class PDFBatchProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.language_detector = PDFLanguageDetector()
        self.ocr_processor = PDFOCRProcessor(self.language_detector)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_single_pdf(self, filename: str) -> Optional[Dict]:
        """Process a single PDF file with error handling"""
        try:
            start_time = time.time()
            filepath = os.path.join(self.input_dir, filename)
            
            converter = EnhancedPDFToOutline(
                language_detector=self.language_detector,
                ocr_processor=self.ocr_processor
            )
            
            result = converter.convert_pdf(filepath)
            output_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": result["title"],
                    "outline": result["outline"],
                    "pdf_type": result.get("pdf_type", "unknown"),
                    "languages": result.get("languages", []),
                    "processing_time": time.time() - start_time,
                    "error": None
                }, f, indent=2, ensure_ascii=False)
            
            return {
                "filename": filename,
                "status": "success",
                "time": time.time() - start_time
            }
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return {
                "filename": filename,
                "status": "failed",
                "error": str(e)
            }

    def process_batch(self, max_workers: int = 8) -> Dict[str, float]:
        """Process all PDFs in directory with parallel execution"""
        start_time = time.time()
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        stats = {"success": 0, "failed": 0, "total": len(pdf_files)}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_pdf, f) for f in pdf_files]
            for future in as_completed(futures):
                result = future.result()
                if result and result["status"] == "success":
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                if result:
                    print(f"Processed {result['filename']} in {result.get('time', 0):.2f}s")
        
        stats["total_time"] = time.time() - start_time
        stats["avg_time"] = stats["total_time"] / max(1, stats["success"])
        
        # Save summary
        with open(os.path.join(self.output_dir, "processing_summary.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    processor = PDFBatchProcessor(input_folder, output_folder)
    stats = processor.process_batch()
    
    print(f"\nProcessing complete!")
    print(f"Total files: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Average time per file: {stats['avg_time']:.2f} seconds")