import os
import cv2
import pytesseract
from PIL import Image
import google.generativeai as genai
import json

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Update this path as per your Tesseract installation

# Configure Gemini API
genai.configure(api_key="AIzaSyDM7R2lwIkxpoN-LufRrgqE5T5BaVv5pE4")

def extract_text_from_image(image_path):
    """Extract raw text from image using Tesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    print("===== OCR Output =====")
    print(text)
    print("======================")
    return text

def extract_face(image_path, output_path="extracted_photo.jpg"):
    """Extract face from image and save it."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        cv2.imwrite(output_path, face)
        return output_path
    return None

def parse_with_gemini(doc_type, raw_text):
    """Send OCR text to Gemini for structured extraction."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    fields_map = {
        "Aadhaar Card": ["Aadhaar Number", "Name", "DOB", "Gender", "Address"],
        "PAN Card": ["PAN Number", "Name", "Father's Name", "DOB"],
        "Driving License": ["DL Number", "Name", "DOB", "Blood Group", "Issue Date", "Valid Till"]
    }

    prompt = f"""
    You are an expert in reading Indian ID documents.
    Extract the details for this {doc_type} from the provided OCR text.

    OCR TEXT:
    {raw_text}

    Return the result as JSON with ONLY these fields: {', '.join(fields_map[doc_type])}.
    If a field is not found, use "Not Found".
    """

    response = model.generate_content(prompt)

    try:
        clean_text = response.text.strip().strip("`").replace("json", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print("Error parsing Gemini response:", e)
        return {"error": "Failed to parse Gemini output", "raw": response.text}
