# Streamlit Auto-Fill Document Extractor

## Project Overview
This project is a **Streamlit-based document extractor** that automates the extraction of key details from Indian identity documents using **OCR** and **Google Gemini API**.  
It supports **Aadhaar Card, PAN Card, and Driving License**, extracting structured data and displaying it in a clean, user-friendly interface.

Key features:
- Upload images of Aadhaar, PAN, or Driving License
- Extract text using **Tesseract OCR**
- Detect and crop face from document images
- Parse text with Gemini 1.5 model for structured output
- Display results alongside the extracted photo
- Handles missing fields gracefully

---

## Features
- **Multi-document support:** Aadhaar, PAN, and Driving License
- **OCR Extraction:** Extract raw text from images using Tesseract
- **Face Detection:** Automatically detects and crops faces
- **AI-based Parsing:** Gemini API generates structured JSON output
- **Streamlit UI:** Clean interface with separate columns for photo and extracted data
- **Error Handling:** Missing fields show "Not Found"

---

## Setup
1. Enter your GeminiAI API KEY 
2. Make sure you install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
3. During the installation setup , choose anyone in the computer have access to tesseract
4. Go to Environment Variable --> System Variables --> Path --> Edit --> Enter your Tesseract Path --> Save
5. Now put the same path in the ocr3.py code at the beginning in the desired line.
6. Run=  streamlit run app3.py