# CSV Dataset Question-Answering System using Gemini API

## Project Overview
This project allows users to interactively ask questions about a CSV dataset using **Google Gemini API**.  
It acts as a **data assistant**, providing clear answers based on the dataset content.

Key features include:
- Load and inspect CSV datasets
- Generate AI-driven answers to dataset-related questions
- Interactive command-line interface
- Supports datasets of varying sizes by sending a snippet to Gemini

---

## Features
- Load CSV files using **pandas**
- Preview dataset columns, row count, and sample data
- Ask questions interactively about the dataset
- Use Gemini 1.5 model to answer questions based on a sample of the dataset
- Handles user exit gracefully

---

## Setup
1. Set your GEMINIAI API KEY
2. Make sure the dataset csvpath is correct
3. Run python qna.py
