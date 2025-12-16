import pytesseract
import cv2
import re
from PIL import Image
import os

# Set tesseract path if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this if needed

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def extract_face(image_path, output_path="extracted_photo.jpg"):
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

def parse_aadhaar_data(text):
    data = {}
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Join lines to search for Aadhaar
    full_text = " ".join(lines)

    # Aadhaar Number
    aadhaar = re.search(r"\d{4}\s\d{4}\s\d{4}", full_text)
    data["Aadhaar Number"] = aadhaar.group() if aadhaar else "Not Found"

    # DOB
    dob_match = re.search(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b", full_text)
    data["DOB"] = dob_match.group() if dob_match else "Not Found"

    # Gender
    gender_match = re.search(r"\b(MALE|FEMALE|OTHER|Male|Female|Other)\b", full_text, re.IGNORECASE)
    data["Gender"] = gender_match.group().capitalize() if gender_match else "Not Found"

    # Name (line before DOB)
    name = "Not Found"
    for i, line in enumerate(lines):
        if "DOB" in line or re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", line):
            if i > 0:
                potential_name = lines[i - 1]
                if re.match(r"^[A-Za-z .]{4,}$", potential_name):  # Name format
                    name = potential_name.strip()
            elif re.match(r"^[A-Za-z .]{4,}$", line):
                name = line.strip()
            break
    data["Name"] = name

    # Address extraction (improved)    # Address extraction
    address = []
    address_keywords = ["Address", "S/O", "C/O", "D/O", "W/O", "H/O"]
    address_started = False
    for line in lines:
        if any(keyword in line for keyword in address_keywords):
            address_started = True
            address.append(line)
        elif address_started:
            # Break if we hit any end indicators
            if (
                re.search(r"\d{4}\s\d{4}\s\d{4}", line)  # Aadhaar number
                or "DOB" in line
                or re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", line)  # DOB
                or re.search(r"\b(MALE|FEMALE|OTHER)\b", line, re.IGNORECASE)  # Gender
                or re.search(r"\bMobile[:\s]*\d{10}\b", line)  # Mobile with label
                or re.search(r"\b\d{10}\b", line)  # Just a 10-digit number
            ):
                break
            address.append(line)

    data["Address"] = " ".join(address) if address else "Not Found"

    return data
