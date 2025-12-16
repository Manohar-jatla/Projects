import streamlit as st
from ocr_aadhar import extract_text_from_image, extract_face, parse_aadhaar_data
import os

st.set_page_config(page_title="Auto-Fill Aadhaar Extractor", layout="centered")

st.title("🔍 Aadhaar Auto-Fill Extractor")
uploaded_file = st.file_uploader("Upload Aadhaar Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting details..."):
        text = extract_text_from_image(image_path)
        extracted_data = parse_aadhaar_data(text)
        face_path = extract_face(image_path)

    st.success("✅ Extraction complete!")

    # 2:1 layout - image right, data left
    col2, col1 = st.columns([2, 1])

    # Show face with vertical spacing above
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical space (~2cm visually)
        if face_path and os.path.exists(face_path):
            st.image(face_path, caption="Extracted Photo", width=120)
        else:
            st.warning("No face detected.")

    # Show Aadhaar extracted info
    with col2:
        st.markdown("### Extracted Information:")
        st.text_input("Name", value=extracted_data["Name"])
        st.text_input("Date of Birth", value=extracted_data["DOB"])
        st.text_input("Gender", value=extracted_data["Gender"])
        st.text_area("Address", value=extracted_data["Address"], height=150)
