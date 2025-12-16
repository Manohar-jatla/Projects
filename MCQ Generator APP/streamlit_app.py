import streamlit as st
import requests
import random
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MCQ Generator & Shuffler", layout="wide")
st.title("📘 MCQ Generator from PDF with Shuffle & Download")
st.info("👋 App loaded! Use the sidebar to begin.")

# --- Initialize session state ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "original_mcqs" not in st.session_state:
    st.session_state.original_mcqs = []
if "shuffled_mcqs" not in st.session_state:
    st.session_state.shuffled_mcqs = []

# --- Sidebar Inputs ---
st.sidebar.header("📝 Input Parameters")

# 🔼 Upload PDF
with st.sidebar.form("upload_form"):
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    submitted = st.form_submit_button("Upload")
    if submitted and uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(f"{API_URL}/upload-doc", files=files)
        if res.status_code == 200:
            st.session_state.doc_id = res.json().get("doc_id")
            st.success(f"Uploaded: {st.session_state.doc_id}")
        else:
            st.error("Upload failed.")

# 🔁 List existing uploaded documents
doc_list_response = requests.get(f"{API_URL}/documents")
doc_options = []

if doc_list_response.status_code == 200:
    documents = doc_list_response.json().get("documents", [])
    doc_options = [doc["doc_id"] for doc in documents]

selected_doc_id = st.sidebar.selectbox("📄 Use Existing Document", options=[""] + doc_options)

if selected_doc_id:
    st.session_state.doc_id = selected_doc_id
    st.sidebar.success(f"Selected existing doc: {selected_doc_id}")

# 🧠 Difficulty & Count
difficulty = st.sidebar.selectbox("Select Difficulty", ["easy", "medium", "hard"])
num_questions = st.sidebar.number_input("Number of Questions", min_value=1, value=5)

# --- Generate MCQs ---
if st.sidebar.button("🔍 Generate MCQs"):
    if not st.session_state.doc_id:
        st.warning("Upload or select a document first.")
    else:
        with st.spinner("Generating MCQs..."):
            payload = {
                "doc_id": st.session_state.doc_id,
                "difficulty": difficulty,
                "num_questions": str(num_questions),
            }
            res = requests.post(f"{API_URL}/generate-mcqs/from-doc", data=payload)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict) and "questions" in data:
                    st.session_state.original_mcqs = data["questions"]
                    st.session_state.shuffled_mcqs = []
                    st.success("✅ MCQs generated!")
                elif isinstance(data, list):
                    st.session_state.original_mcqs = data
                    st.session_state.shuffled_mcqs = []
                    st.success("✅ MCQs generated!")
                else:
                    st.error("❌ Unexpected format from API.")
                    st.json(data)
            else:
                st.error(f"❌ Failed to generate MCQs. ({res.status_code})")
                st.text(res.text)

# --- Shuffle Questions ---
if st.button("🔁 Shuffle Questions"):
    if not st.session_state.original_mcqs:
        st.warning("Generate original MCQs first.")
    else:
        new_shuffle = random.sample(st.session_state.original_mcqs, len(st.session_state.original_mcqs))
        if len(st.session_state.shuffled_mcqs) >= 3:
            st.session_state.shuffled_mcqs.pop(0)
        st.session_state.shuffled_mcqs.append(new_shuffle)
        scroll_target = f"#🔀-shuffled-version-{len(st.session_state.shuffled_mcqs)}"
        st.markdown(f"✅ Questions shuffled! [🔽 Jump to Shuffled Version {len(st.session_state.shuffled_mcqs)}]({scroll_target})")


# --- MCQ Display Function ---
def display_mcqs(title, questions):
    anchor = title.replace(" ", "-").lower()
    st.markdown(f'<a name="{anchor}"></a>', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if not isinstance(questions, list):
        st.error("❌ Unexpected MCQ format (not a list).")
        st.text(str(questions))
        return

    for i, q in enumerate(questions, 1):
        if isinstance(q, dict):
            st.markdown(f"**Q{i}. {q.get('question', 'No question text')}**")
            options = q.get("options", [])
            for opt in options:
                st.markdown(f"- {opt}")
            st.markdown(f"**Answer**: {q.get('answer', 'N/A')}")
            if q.get("reference"):
                st.markdown(f"📌 *Reference*: {q['reference']}")
            st.markdown("---")
        else:
            st.warning(f"⚠️ MCQ {i} not in expected format. Skipped.")

# --- Display All MCQs ---
if st.session_state.original_mcqs:
    display_mcqs("📘 Original MCQs", st.session_state.original_mcqs)

for idx, shuffle in enumerate(st.session_state.shuffled_mcqs):
    display_mcqs(f"🔀 Shuffled Version {idx + 1}", shuffle)

# --- Download Section ---
if st.session_state.original_mcqs or st.session_state.shuffled_mcqs:
    st.markdown("## 📥 Download MCQs")

    version_options = ["Original"] + [f"Shuffle {i + 1}" for i in range(len(st.session_state.shuffled_mcqs))]
    selected = st.selectbox("Choose a version to download", version_options)

    if st.button("⬇️ Download Selected Version"):
        selected_data = (
            st.session_state.original_mcqs
            if selected == "Original"
            else st.session_state.shuffled_mcqs[int(selected.split()[-1]) - 1]
        )
        lines = []
        for i, q in enumerate(selected_data, 1):
            lines.append(f"Q{i}: {q.get('question', 'N/A')}")
            lines.append("Options: " + ", ".join(q.get("options", [])))
            lines.append("Answer: " + q.get("answer", 'N/A'))
            if q.get("reference"):
                lines.append("Reference: " + q["reference"])
            lines.append("")

        txt_data = "\n".join(lines)
        st.download_button(
            label="📥 Download as .txt",
            data=txt_data,
            file_name=f"{selected.replace(' ', '_').lower()}_mcqs.txt",
            mime="text/plain"
        )
