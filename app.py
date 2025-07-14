import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔐 Configure Gemini
genai.configure(api_key=st.secrets["gemini"]["api_key"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 🧠 Embedding model for semantic similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🔍 AI Job Finder", layout="wide")
st.title("🔍 AI-Powered CV Matcher & Real-Time Job Finder")
st.markdown("Upload your CV to get real job matches from Adzuna, with AI cover letter help.")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

# -------------------- PDF Text Extraction --------------------
def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# -------------------- Adzuna API Job Search --------------------
def fetch_jobs_from_adzuna(query, location="United States", country="us", max_results=100):
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": st.secrets["adzuna"]["app_id"],
        "app_key": st.secrets["adzuna"]["app_key"],
        "results_per_page": max_results,
        "what": query,
        "where": location,
        "content-type": "application/json"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"❌ Adzuna API Error: {response.text}")
        return []
    jobs = response.json().get("results", [])
    return [
        {
            "title": job["title"],
            "description": job.get("description", ""),
            "location": job.get("location", {}).get("display_name", "")
        }
        for job in jobs
    ]

# -------------------- Main App Logic --------------------
if uploaded_file:
    with st.spinner("📄 Extracting and analyzing your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)

        # 🎯 Get summary from Gemini
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
        try:
            response = model.generate_content(prompt)
            cv_summary = response.text
        except Exception as e:
            st.error(f"❌ Gemini Error: {e}")
            st.stop()

        st.subheader("🧾 AI-Curated CV Summary")
        st.info(cv_summary)

        # 🔍 Extract keywords
        keyword_prompt = f"""
        Based on this CV, extract 1–3 job search keywords (e.g. 'data analyst', 'marketing intern').

        CV:
        {cv_summary}
        """
        try:
            keywords_resp = model.generate_content(keyword_prompt).text
            st.subheader("🧠 Suggested Job Search Keywords")
            st.code(keywords_resp.strip())
        except:
            keywords_resp = "data analyst"

        # Manual override for keywords and location
        search_keywords = st.text_input("🔍 Edit search keywords", value=keywords_resp.strip().split("\n")[0] if keywords_resp else "data analyst")
        location = st.text_input("📍 Location", value="New Zealand")

        # 🔎 Fetch jobs from Adzuna
        st.subheader(f"🌐 Searching Adzuna for: `{search_keywords}` in `{location}`")
        jobs = fetch_jobs_from_adzuna(query=search_keywords, location=location)

        if not jobs:
            st.error("❌ No jobs found. Try different keywords or a broader location.")
            st.write("🔎 Keywords used:", search_keywords)
            st.write("📍 Location used:", location)
            st.stop()

        # 🧠 Match jobs using semantic similarity
        st.subheader("✅ Top Matched Jobs")
        with st.spinner("🔎 Matching your CV with job descriptions..."):
            cv_vector = embedder.encode([cv_summary])[0]
            match_scores = []
            for job in jobs:
                job_vector = embedder.encode([job["description"]])[0]
                score = cosine_similarity([cv_vector], [job_vector])[0][0]
                match_scores.append((job, score))

            top_jobs = sorted(match_scores, key=lambda x: x[1], reverse=True)[:3]

        for job, score in top_jobs:
            st.markdown(f"**🧑‍💼 {job['title']}** in *{job['location']}* — Match Score: `{round(score*100, 2)}%`")
            st.markdown(f"> {job['description'][:400]}...")
            st.markdown("---")

        # 📝 Generate cover letter advice
        st.subheader("✍️ Gemini-Powered Application Advice")
        best_job = top_jobs[0][0]
        advice_prompt = f"""
        I am applying for this job: {best_job['title']}
        Job description: {best_job['description']}

        My CV summary is: {cv_summary}

        What should I highlight in my application?
        Can you write a short, tailored cover letter?
        """
        try:
            letter = model.generate_content(advice_prompt).text
            st.success(letter)
        except Exception as e:
            st.error(f"❌ Cover letter generation failed: {e}")
