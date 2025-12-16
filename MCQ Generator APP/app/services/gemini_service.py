import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Explicitly load .env from the root directory (2 levels up)
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")


def generate_mcqs(text: str, difficulty: str, num_questions: int):
    prompt = (
        f"Generate {num_questions} {difficulty} level multiple-choice questions from the text below:\n\n{text}\n\n"
        f"Each MCQ should follow this exact format:\n"
        f"**Question:** <question text>\n"
        f"a) <option a>\n"
        f"b) <option b>\n"
        f"c) <option c>\n"
        f"d) <option d>\n"
        f"**Answer:** <correct option letter>\n"
        f"**Explanation:** <short explanation>\n\n"
        f"Return all questions clearly with consistent formatting."
    )

    response = model.generate_content(prompt)

    # Raw Gemini text output
    raw = response.text.strip()

    # Split into individual questions using a pattern
    blocks = re.split(r"\*\*Question:\*\*", raw)
    mcqs = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        try:
            # Extract question
            question_match = re.match(r"(.*?)(?:\na\)|\na\.)", block, re.DOTALL)
            question_text = question_match.group(1).strip() if question_match else "Question not found"

            # Extract all options
            options = re.findall(r"[a-d]\)\s*(.*)", block)

            # Extract answer
            answer_match = re.search(r"\*\*Answer:\*\*\s*([a-dA-D])", block)
            answer_letter = answer_match.group(1).lower() if answer_match else "a"
            answer = f"{answer_letter}) {options[ord(answer_letter) - ord('a')]}" if options else "a) Missing option"

            # Extract explanation
            explanation_match = re.search(r"\*\*Explanation:\*\*\s*(.*)", block, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided."

            mcqs.append({
                "question": question_text,
                "options": [f"{chr(97+i)}) {opt.strip()}" for i, opt in enumerate(options)],
                "answer": answer,
                "explanation": explanation
            })

        except Exception as e:
            print("❌ Parsing failed for block:\n", block)
            print("Error:", e)

    return mcqs
