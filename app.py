import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 🔐 Gemini setup
genai.configure(api_key=st.secrets["gemini"]["api_key"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 🧠 SentenceTransformer for semantic similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🌍 Country codes for Adzuna
country_map = {
    "New Zealand": "nz",
    "Australia": "au",
    "United States": "us",
    "United Kingdom": "gb",
    "Canada": "ca",
    "India": "in",
    "Germany": "de",
    "France": "fr",
    "Netherlands": "nl",
    "South Africa": "za"
}

# -------------------- Adzuna API Job Fetch --------------------
def fetch_jobs_from_adzuna(query, location="New York", country="us", max_results=10):
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

        # 🔍 Keyword Extraction and Cleaning
        keyword_prompt = f"""
        Based on this CV, extract the top 1–3 job search keywords (e.g. 'professor', 'marketing analyst').
        Return just the keywords.
        CV:
        {cv_summary}
        """
        try:
            raw_keywords = model.generate_content(keyword_prompt).text
            st.subheader("🧠 Raw AI Keywords")
            st.code(raw_keywords.strip())
        except:
            raw_keywords = "professor"

        first_line = raw_keywords.strip().split("\n")[0]
        clean_keyword = re.sub(r"[^a-zA-Z0-9\s]", "", first_line)
        clean_keyword = re.sub(r"^\d+\s*", "", clean_keyword).strip().lower()

        # 🔧 User override
        search_keywords = st.text_input("🔍 Edit search keywords", value=clean_keyword or "professor")

        country_name = st.selectbox("🌍 Country", list(country_map.keys()), index=2)  # Default: US
        country_code = country_map[country_name]

        location = st.text_input("📍 City or Region", value="New York")

        st.subheader(f"🌐 Searching Adzuna for: `{search_keywords}` in `{location}`, {country_name}")
        jobs = fetch_jobs_from_adzuna(query=search_keywords, location=location, country=country_code)

        if not jobs:
            st.error("❌ No jobs found. Try a different keyword or broader location.")
            st.write("🔎 Keyword used:", search_keywords)
            st.write("📍 Location used:", location)
            st.write("🌍 Country used:", country_code)
            st.stop()

        # 🧠 Semantic Matching
        with st.spinner("🔎 Matching your CV with job descriptions..."):
            cv_vector = embedder.encode([cv_summary])[0]
            match_scores = []
            for job in jobs:
                job_vector = embedder.encode([job["description"]])[0]
                score = cosine_similarity([cv_vector], [job_vector])[0][0]
                match_scores.append((job, score))

            top_jobs = sorted(match_scores, key=lambda x: x[1], reverse=True)[:3]

        st.subheader("✅ Top Matched Jobs")
        for job, score in top_jobs:
            st.markdown(f"**🧑‍💼 {job['title']}** in *{job['location']}* — Match Score: `{round(score*100, 2)}%`")
            st.markdown(f"> {job['description'][:400]}...")
            st.markdown("---")

        # ✍️ Gemini-Powered Cover Letter
        st.subheader("✍️ Gemini AI Cover Letter Advice")
        best_job = top_jobs[0][0]
        advice_prompt = f"""
        I am applying for this job: {best_job['title']}
        Job description: {best_job['description']}

        My CV summary is: {cv_summary}

        What should I highlight in my application?
        Can you help me write a short, tailored cover letter?
        """
        try:
            letter = model.generate_content(advice_prompt).text
            st.success(letter)
        except Exception as e:
            st.error(f"❌ Cover letter generation failed: {e}")
