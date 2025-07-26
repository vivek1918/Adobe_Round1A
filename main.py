import os
import json
import time
from pdf_outline_converter import EnhancedPDFToOutline

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
    input_folder = "input"
    output_folder = "output"
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    process_pdfs(input_folder, output_folder)