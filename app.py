import streamlit as st
import fitz
import requests
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import json
from io import BytesIO

# -------------------- Neon CSS Styling --------------------
st.markdown("""
    <style>
    .neon-box {
        border: 2px solid #00FFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px #00FFFF;
        background-color: #0f0f0f;
    }
    .stTextInput > div > div {
        border-color: #00FFFF;
    }
    .stApp {
        background-color: #000000;
        color: white;
    }
    .css-ffhzg2 { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# -------------------- Twilio WhatsApp --------------------
try:
    twilio_sid = st.secrets["twilio"]["account_sid"]
    twilio_token = st.secrets["twilio"]["auth_token"]
    whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
    whatsapp_from = "whatsapp:+14155238886"
except KeyError:
    st.error("🔐 Missing Twilio credentials in `.streamlit/secrets.toml`.")
    st.stop()

def send_whatsapp_alert(message):
    from twilio.rest import Client
    client = Client(twilio_sid, twilio_token)
    client.messages.create(body=message, from_=whatsapp_from, to=whatsapp_to)

# -------------------- DeepSeek Setup --------------------
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
        "messages": [{"role": "system", "content": "You are a helpful AI assistant."},
                     {"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"❌ DeepSeek error: {response.text}")
        return "Sorry, DeepSeek failed to respond."

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- Adzuna Job Fetch --------------------
country_map = {
    "New Zealand": "nz", "Australia": "au", "United States": "us",
    "United Kingdom": "gb", "Canada": "ca", "India": "in"
}
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
    return [
        {
            "title": job["title"],
            "description": job.get("description", ""),
            "location": job.get("location", {}).get("display_name", ""),
            "url": job.get("redirect_url", "")
        }
        for job in response.json().get("results", [])
    ]

# -------------------- PDF Extraction --------------------
def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def generate_pdf(text, filename="tailored_cv.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="💡 AI CV Matcher", layout="wide")
st.title("🚀 AI CV Matcher with WhatsApp Alert, PDF Download, and Neon Demo UI")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Analyzing your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)

        cv_summary = ask_deepseek(f"Summarize this CV: {cv_text}")
        job_keywords = ask_deepseek(f"Extract top job roles for this CV: {cv_summary}").split("\n")[0]
        clean_keyword = re.sub(r"[^a-zA-Z0-9\s]", "", job_keywords).strip().lower()

    st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b> {clean_keyword.title()}</div>', unsafe_allow_html=True)
    country_name = st.selectbox("🌍 Choose Country", list(country_map.keys()), index=2)
    country_code = country_map[country_name]
    location = st.text_input("📍 City or Region (optional)", "")

    jobs = fetch_jobs_from_adzuna(query=clean_keyword, location=location, country=country_code)
    if not jobs:
        st.error("❌ No jobs found.")
        st.stop()

    with st.spinner("🔍 Matching CV with jobs..."):
        cv_vector = embedder.encode([cv_summary])[0]
        match_scores = []
        for job in jobs:
            job_vector = embedder.encode([job["description"]])[0]
            score = cosine_similarity([cv_vector], [job_vector])[0][0]
            match_scores.append((job, score))
        top_jobs = sorted(match_scores, key=lambda x: x[1], reverse=True)

    st.subheader("📊 Top Matches")

    for i, (job, score) in enumerate(top_jobs):
        match_percent = round(score * 100, 2)
        st.markdown(f"### 🔹 [{job['title']} - {job['location']}]({job['url']}) — {match_percent}% Match")
        with st.expander("📝 View Details"):
            st.markdown(job["description"])

            advice = ask_deepseek(f"Evaluate this job: {job['title']}\n{job['description']}\nCV: {cv_summary}")
            st.success(advice)

            tailoring_prompt = f"Write a full tailored CV based on this job description: {job['description']}\nOriginal CV: {cv_summary}"
            if st.button("✍️ Generate Tailored CV", key=f"cv_{i}"):
                tailored_cv = ask_deepseek(tailoring_prompt)
                st.text_area("📄 Tailored CV (Formatted)", tailored_cv, height=400)

                pdf_file = generate_pdf(tailored_cv)
                st.download_button("📥 Download as PDF", data=pdf_file, file_name="Tailored_CV.pdf")

                st.button("📧 Do you want to email your tailored CV and cover letter?", key=f"email_dummy_{i}")

            if score >= 0.5:
                try:
                    send_whatsapp_alert(f"📬 Job Match Alert!\n{job['title']} ({match_percent}%)\n{job['url']}")
                    st.success("📲 WhatsApp alert sent!")
                except Exception as e:
                    st.warning(f"❌ WhatsApp alert failed: {str(e)}")

    # 🧠 CV Quality Score
    st.subheader("📈 CV Quality Score (AI)")
    score_prompt = f"Give a CV quality score from 0–100 and explain reasoning for this CV:\n{cv_summary}"
    score_feedback = ask_deepseek(score_prompt)
    st.markdown(f'<div class="neon-box">{score_feedback}</div>', unsafe_allow_html=True)

    # ❓ AI Q&A Section
    st.subheader("🤖 Ask Anything About Your Career or CV")
    user_q = st.text_input("💬 Ask a question to the AI:")
    if user_q:
        ai_a = ask_deepseek(f"Q: {user_q}\nContext: {cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {ai_a}")
