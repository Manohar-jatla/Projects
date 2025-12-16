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

# Function to generate and format MCQs
def generate_mcqs(text: str, difficulty: str, num_questions: int):
    prompt = (
        f"Generate {num_questions} {difficulty} level multiple-choice questions "
        f"from the following text:\n\n{text}\n\n"
        f"Each question should have 4 options (a, b, c, d), clearly indicate the correct answer, "
        f"and provide a short explanation or reference after each question."
    )
    
    response = model.generate_content(prompt)
    
    # Split into questions at **Question 1:**, **Question 2:**, etc.
    questions = re.split(r"\*\*Question \d+:\*\*", response.text)
    
    # Clean and return non-empty parts
    cleaned = [q.strip() for q in questions if q.strip()]
    
    return {"questions": cleaned}
