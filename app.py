import streamlit as st
import fitz  # PyMuPDF
import requests
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# ----------------- DeepSeek via OpenRouter Setup -----------------
OPENROUTER_API_KEY = st.secrets["openrouter"]["api_key"]
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://chat.openai.com/",
    "Content-Type": "application/json"
}

def ask_deepseek(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an AI job assistant helping analyze CVs and job postings."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"❌ DeepSeek error: {response.text}")
        return "Sorry, DeepSeek failed to respond."

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

        summary_prompt = f"""
        Extract the following from this CV:
        - Top skills
        - Work experience
        - Education
        - Job roles or titles held
        - Career interests (if mentioned)

        CV Text:
        {cv_text}
        """
        cv_summary = ask_deepseek(summary_prompt)

        st.subheader("🧾 AI-Curated CV Summary")
        st.info(cv_summary)

        keyword_prompt = f"""
        Based on this CV, extract the top 1–3 job search keywords (e.g. 'data analyst', 'AI engineer').
        Return just the keywords as a plain list.
        CV:
        {cv_summary}
        """
        raw_keywords = ask_deepseek(keyword_prompt)
        st.subheader("🧠 Best Role Suited for You")
        st.code(raw_keywords.strip())

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
        for i, (job, score) in enumerate(top_jobs):
            st.markdown(
                f"### [{job['title']} - {job['location']}]({job['url']})\n"
                f"**Match:** {round(score * 100, 2)}%",
                help="Click to view job details and tailored advice"
            )
            with st.expander("🔍 View Details"):
                st.markdown(f"**Full Description:**\n\n{job['description']}")

                if st.button(f"✍️ Tailor CV & Cover Letter for {job['title']}", key=f"tailor_{i}"):
                    tailoring_prompt = f"""
                    Based on my CV summary below, tailor a short professional CV summary and a 1-paragraph cover letter
                    for this job titled: {job['title']}.

                    Job description: {job['description']}
                    My CV summary: {cv_summary}
                    """
                    tailored_docs = ask_deepseek(tailoring_prompt)
                    st.text_area("📄 Tailored Summary + Cover Letter", tailored_docs, height=250)

                emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', job["description"])
                if emails:
                    email_to = emails[0]
                    st.markdown(f"📧 **Detected contact email:** `{email_to}`")
                    if st.button("📤 Send tailored CV to email", key=f"send_{i}"):
                        st.warning("⚠️ Email sending not yet enabled. You can integrate SendGrid or SMTP.")
                else:
                    st.info("📭 No contact email found in this job description.")
