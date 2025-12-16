# FastAPI MCQ Generator

This project provides a REST API to generate multiple-choice questions (MCQs) from PDF content using Gemini 1.5 models.

## Features
- Extract text from PDF files
- Generate MCQs with options, correct answers, and references
- Cache generated MCQs for faster repeat access
- Easy integration with frontend via FastAPI endpoints

## Setup
1. Set your GeminiAI API key as environment variable `GEMINI_API_KEY`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn app.main:app --reload`

## API Endpoint
- `POST /generate-mcqs`: Upload a PDF file, specify difficulty and number of questions to generate MCQs

## Example cURL
```bash
curl -X POST "http://localhost:8000/generate-mcqs" -F "file=@path_to_pdf.pdf" -F "difficulty=easy" -F "num_questions=3"
```