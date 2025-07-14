import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔐 Configure Gemini
genai.configure(api_key=st.secrets["gemini"]["api_key"])
model = genai.GenerativeModel("gemini-pro")

# 🧠 Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------- Dummy Job Data -----------------------
JOB_DB = [
    {
        "title": "Data Analyst",
        "description": "Seeking a data analyst proficient in SQL, Python, and data visualization tools. Experience in business analytics and reporting is preferred."
    },
    {
        "title": "AI Research Intern",
        "description": "Assist in building NLP models. Background in Python, ML libraries (Scikit-learn, TensorFlow), and academic research experience a plus."
    },
    {
        "title": "Marketing Analyst",
        "description": "Analyze marketing trends and customer data. Should have Excel, Google Analytics, and dashboarding skills."
    },
    {
        "title": "Junior Software Developer",
        "description": "Looking for an entry-level developer with skills in Python, APIs, and debugging. React or Streamlit experience is a bonus."
    }
]

# ----------------------- App UI -----------------------
st.set_page_config(page_title="🔍 Smart Job Match AI", layout="wide")
st.title("🔍 AI-Powered CV Matcher & Application Advisor")
st.markdown("Upload your CV and find your best job matches with AI-generated application advice.")

# ----------------------- CV Upload & Extract -----------------------
uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

if uploaded_file:
    with st.spinner("📄 Reading and analyzing your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)

        # ----------------------- Gemini Summary -----------------------
        prompt = f"""
        Extract the following from this CV text:
        - Top skills
        - Work experience
        - Education
        - Job roles or titles held
        - Career interests (if mentioned)
        
        CV Text:
        {cv_text}
        """
        cv_summary = model.generate_content(prompt).text
        st.subheader("🧾 CV Summary (AI-Extracted)")
        st.info(cv_summary)

        # ----------------------- Job Matching -----------------------
        with st.spinner("🔍 Matching with available jobs..."):
            cv_vector = embedder.encode([cv_summary])[0]

            match_scores = []
            for job in JOB_DB:
                job_vector = embedder.encode([job["description"]])[0]
                score = cosine_similarity([cv_vector], [job_vector])[0][0]
                match_scores.append((job, score))

            top_jobs = sorted(match_scores, key=lambda x: x[1], reverse=True)[:3]

            st.subheader("✅ Best Job Matches")
            for job, score in top_jobs:
                st.markdown(f"**🧑‍💼 {job['title']}** (Match Score: `{round(score*100, 2)}%`)")
                st.markdown(f"> {job['description']}")
                st.markdown("---")

            # ----------------------- Application Advice -----------------------
            best_job = top_jobs[0][0]
            advice_prompt = f"""
            I am applying for this job: {best_job['title']}
            Job description: {best_job['description']}
            
            My CV summary is: {cv_summary}

            What should I highlight in my application?
            Can you suggest how to write a tailored cover letter for this role?
            """
            st.subheader("📝 Cover Letter & Application Advice")
            cover_letter_advice = model.generate_content(advice_prompt).text
            st.success(cover_letter_advice)
