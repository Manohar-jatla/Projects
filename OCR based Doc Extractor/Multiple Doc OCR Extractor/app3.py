import os
import streamlit as st
from ocr3 import extract_text_from_image, extract_face, parse_with_gemini

st.set_page_config(page_title="Auto-Fill Document Extractor", layout="centered")
st.title("🧾 Auto-Fill Document Extractor")

# Document selection
doc_type = st.selectbox("Select Document Type", ["Aadhaar Card", "PAN Card", "Driving License"])
uploaded_file = st.file_uploader(f"Upload {doc_type} Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting details..."):
        text = extract_text_from_image(image_path)
        extracted_data = parse_with_gemini(doc_type, text)
        face_path = extract_face(image_path)

    st.success("✅ Extraction complete!")

    # Display results in columns
    col_data, col_img = st.columns([2, 1])

    with col_img:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical space (~2cm visually)
        if face_path and os.path.exists(face_path):
            st.image(face_path, caption="Extracted Photo", width=120)
        else:
            st.warning("No face detected.")

    with col_data:
        st.markdown("### Extracted Information:")

        if isinstance(extracted_data, dict) and "error" not in extracted_data:
            # Display only fields relevant to the selected doc_type
            if doc_type == "Aadhaar Card":
                st.text_input("Aadhaar Number", value=extracted_data.get("Aadhaar Number", "Not Found"))
                st.text_input("Name", value=extracted_data.get("Name", "Not Found"))
                st.text_input("DOB", value=extracted_data.get("DOB", "Not Found"))
                st.text_input("Gender", value=extracted_data.get("Gender", "Not Found"))
                st.text_area("Address", value=extracted_data.get("Address", "Not Found"))

            elif doc_type == "PAN Card":
                st.text_input("PAN Number", value=extracted_data.get("PAN Number", "Not Found"))
                st.text_input("Name", value=extracted_data.get("Name", "Not Found"))
                st.text_input("Father's Name", value=extracted_data.get("Father's Name", "Not Found"))
                st.text_input("DOB", value=extracted_data.get("DOB", "Not Found"))

            elif doc_type == "Driving License":
                st.text_input("DL Number", value=extracted_data.get("DL Number", "Not Found"))
                st.text_input("Name", value=extracted_data.get("Name", "Not Found"))
                st.text_input("DOB", value=extracted_data.get("DOB", "Not Found"))
                st.text_input("Blood Group", value=extracted_data.get("Blood Group", "Not Found"))
                st.text_input("Issue Date", value=extracted_data.get("Issue Date", "Not Found"))
                st.text_input("Valid Till", value=extracted_data.get("Valid Till", "Not Found"))

        else:
            st.error("Could not parse extracted data.")
