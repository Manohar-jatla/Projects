from pydantic import BaseModel
from typing import List

class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

class MCQResponse(BaseModel):
    questions: List[MCQ]
