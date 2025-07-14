import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import requests
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 🔐 Gemini setup
genai.configure(api_key=st.secrets["gemini"]["api_key"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 🧠 SentenceTransformer for semantic similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🌍 Country codes for Adzuna
country_map = {
    "New Zealand": "nz", "Australia": "au", "United States": "us",
    "United Kingdom": "gb", "Canada": "ca", "India": "in",
    "Germany": "de", "France": "fr", "Netherlands": "nl", "South Africa": "za"
}

# -------------------- Adzuna API Job Fetch --------------------
def fetch_jobs_from_adzuna(query, location="", country="us", max_results=10):
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
            "location": job.get("location", {}).get("display_name", ""),
            "url": job.get("redirect_url", "")
        }
        for job in jobs
    ]

# -------------------- PDF Text Extraction --------------------
def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🌍 Global CV Matcher", layout="wide")
st.title("🔍 AI-Powered Global CV Matcher & Real-Time Job Finder")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Analyzing your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)

        prompt = f"""
        Extract the following from this CV:
        - Top skills
        - Work experience
        - Education
        - Job roles or titles held
        - Career interests (if mentioned)

        CV Text:
        {cv_text}
        """
        try:
            cv_summary = model.generate_content(prompt).text
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

        st.subheader("🧾 Gemini-Curated CV Summary")
        st.info(cv_summary)

        keyword_prompt = f"""
        Based on this CV, extract the top 1–3 job search keywords (e.g. 'data analyst', 'AI engineer').
        Return just the keywords.
        CV:
        {cv_summary}
        """
        try:
            raw_keywords = model.generate_content(keyword_prompt).text
            st.subheader("🧠 Raw AI Keywords")
            st.code(raw_keywords.strip())
        except:
            raw_keywords = "data analyst"

        first_line = raw_keywords.strip().split("\n")[0]
        clean_keyword = re.sub(r"[^a-zA-Z0-9\s]", "", first_line)
        clean_keyword = re.sub(r"^\d+\s*", "", clean_keyword).strip().lower()

        search_keywords = st.text_input("🔍 Edit search keywords", value=clean_keyword or "data analyst")
        country_name = st.selectbox("🌍 Country", list(country_map.keys()), index=2)  # Default: US
        country_code = country_map[country_name]
        location = st.text_input("📍 City or Region (optional)", value="")

        if not location:
            st.warning("🔎 No location specified. Searching broadly in selected country...")

        st.subheader(f"🌐 Searching Adzuna for: `{search_keywords}` in `{location or 'All Regions'}`, {country_name}")
        jobs = fetch_jobs_from_adzuna(query=search_keywords, location=location, country=country_code)

        if not jobs:
            st.error("❌ No jobs found. Try a different keyword or broader location.")
            st.stop()

        with st.spinner("🔎 Matching your CV with job descriptions..."):
            cv_vector = embedder.encode([cv_summary])[0]
            match_scores = []
            for job in jobs:
                job_vector = embedder.encode([job["description"]])[0]
                score = cosine_similarity([cv_vector], [job_vector])[0][0]
                match_scores.append((job, score))

            top_jobs = sorted(match_scores, key=lambda x: x[1], reverse=True)

        st.subheader("📊 Top Job Matches (Semantic Similarity)")
        job_table = pd.DataFrame([
            {
                "Job Title": job["title"],
                "Location": job["location"],
                "Match %": round(score * 100, 2)
            }
            for job, score in top_jobs
        ])
        st.dataframe(job_table, use_container_width=True)

        st.subheader("🧐 Select a job to view full details and advice")
        selected_index = st.selectbox("Choose a job", list(range(len(top_jobs))), format_func=lambda i: top_jobs[i][0]['title'])
        selected_job, selected_score = top_jobs[selected_index]

        st.markdown(f"### 🧾 Full Job Description\n**{selected_job['title']}** in *{selected_job['location']}*")
        st.markdown(selected_job["description"])
        if selected_job["url"]:
            st.markdown(f"🔗 [Apply Here]({selected_job['url']})")

        explain_prompt = f"""
        I am evaluating a job match for this position: {selected_job['title']}
        Job description: {selected_job['description']}

        My CV summary is: {cv_summary}

        - Why is this a good match?
        - What is missing from my CV for this job?
        - Should I apply? Give a short recommendation.
        - If anything is missing, suggest how to improve it.
        """
        try:
            reasoning = model.generate_content(explain_prompt).text
            st.success(reasoning)
        except Exception as e:
            st.error(f"❌ Gemini explanation failed: {e}")
