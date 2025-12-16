# Streamlit PDF MCQ Generator with Shuffling

## Project Overview
This project is an advanced version of the PDF-to-MCQ generator. It provides a **user-friendly Streamlit frontend** where users can:
- Upload a PDF
- Select difficulty level (easy, medium, hard)
- Specify the number of MCQs to generate
- Shuffle the MCQs up to 3 times
- Download the original and shuffled MCQ sets  

The backend uses **Google Gemini 1.5 models** for AI-powered question generation and implements caching to avoid redundant API calls.

---

## Features
- Upload PDFs directly from the Streamlit interface
- Generate multiple-choice questions with options, correct answers, and references
- Specify the number of questions and difficulty level
- Shuffle MCQs and generate up to 3 shuffled versions
- Download original and shuffled MCQ sets as JSON
- Local caching for faster repeat access

---

## Setup
1. Set your GeminiAI API key
2. Run the server= uvicorn app.main:app --reload
3. run the streamlit server= streamlit run streamlit_app.py  

