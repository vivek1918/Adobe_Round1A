# PDF Processing Application

## Overview
This is a Dockerized PDF processing application that performs various operations on PDF documents including OCR, layout analysis, and content extraction.

---

## Prerequisites
- Docker installed on your system  
- Git (for cloning the repository)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/vivek1918/Adobe_Round1A.git
cd Adobe_Round1A

# Build the Docker image
docker build -t pdf-processing-app .

#Usage
Run the application with Docker, mounting your input and output directories: docker run -v "/path/to/your/input:/app/input" -v "/path/to/your/output:/app/output" pdf-processing-app
Replace /path/to/your/input and /path/to/your/output with your local directories containing input PDFs and where you want output to be saved.

#Project Structure
AYEDOBI/
├── input/                  # Directory for input PDF files
│── iopath_cache/
│── output/            # Default output directory (can be overridden by mounted volume)
├── PubLayNet_model/        # Contains layout analysis model
├── venv/                   # Python virtual environment
├── .dockerignore          # Files to exclude from Docker build
├── .gitignore             # Files to exclude from Git
├── Dockerfile             # Docker configuration
├── language_detection.py   # Language detection module
├── load_model.py          # Model loading utilities
├── main.py                # Main application entry point
├── ml.models.py           # Machine learning models
├── ocr_processing.py      # OCR processing module
├── pdf_processing.py      # PDF processing utilities
├── requirements.txt       # Python dependencies
└── utils.py               # Utility functions


#Configuration
Modify the following files as needed:
Dockerfile - To change build configurations
requirements.txt - To update Python dependencies

#Notes
The application expects PDF files in the mounted input directory
Processed files will be saved to the mounted output directory
The PubLayNet_model directory should contain the trained layout analysis model
