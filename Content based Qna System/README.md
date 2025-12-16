# NLTK-based Question Answering System

## Project Overview
This project is a simple Question-Answering (QnA) system built using **Python**, **NLTK**, and **scikit-learn**. 
It allows users to ask questions about a given paragraph, and it returns the most relevant sentence as the answer based on **TF-IDF vectorization** and **cosine similarity**.

The project demonstrates basic Natural Language Processing (NLP) techniques, including:
- Sentence and word tokenization
- Stopword removal
- Word stemming
- TF-IDF vectorization
- Cosine similarity for sentence ranking

---

## Features
- Preprocesses input text using NLTK (tokenization, stopword removal, stemming)
- Accepts a paragraph and a question as input
- Finds the most relevant sentence in the paragraph that answers the question
- Interactive command-line interface

---

## Running
Requires Python 3.12 and older (3.10 is recommended)

python qna_system.py