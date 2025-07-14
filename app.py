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
    </style>
""", unsafe_allow_html=True)

# -------------------- WhatsApp via Twilio --------------------
try:
    twilio_sid = st.secrets["twilio"]["account_sid"]
    twilio_token = st.secrets["twilio"]["auth_token"]
    whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
    whatsapp_from = "whatsapp:+14155238886"
except KeyError:
    st.error("🔐 Missing Twilio credentials.")
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

# -------------------- Dummy Job Fetch --------------------
def fetch_dummy_jobs(keyword, max_results=5):
    dummy_jobs = [
        {"title": f"{keyword.title()} at TechCorp", "description": f"We are hiring a {keyword} to lead AI innovation.", "location": "Remote", "url": "https://example.com/job1"},
        {"title": f"Senior {keyword.title()} Role", "description": f"Looking for an expert in {keyword}.", "location": "New York, USA", "url": "https://example.com/job2"},
        {"title": f"{keyword.title()} Specialist", "description": f"Join our global team as a {keyword}.", "location": "London, UK", "url": "https://example.com/job3"},
    ]
    return dummy_jobs[:max_results]

# -------------------- PDF Handling --------------------
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
st.set_page_config(page_title="🚀 AI CV Matcher", layout="wide")
st.title("🌟 AI-Based CV Matcher with PDF Export, WhatsApp Alerts, and Tailored Applications")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Analyzing your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_summary = ask_deepseek(f"Summarize this CV:\n{cv_text}")

        keyword_prompt = f"""
        From this CV summary, extract the top 3 job roles the candidate is best suited for.
        Return just the roles (e.g., Data Analyst, Machine Learning Engineer). No extra text.
        CV Summary:
        {cv_summary}
        """
        raw_keywords = ask_deepseek(keyword_prompt)
        lines = raw_keywords.strip().split("\n")
        search_keywords = None
        for line in lines:
            clean_line = re.sub(r"[-•0-9]", "", line).strip()
            clean_line = re.sub(r"[^a-zA-Z\s]", "", clean_line).strip()
            if 3 <= len(clean_line) <= 40 and " " in clean_line:
                search_keywords = clean_line.lower()
                break
        if not search_keywords:
            search_keywords = "data analyst"
            st.warning("⚠️ AI failed to extract job role. Using default: 'Data Analyst'.")

        st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b> {search_keywords.title()}</div>', unsafe_allow_html=True)

        jobs = fetch_dummy_jobs(search_keywords)

        st.subheader("📊 Matched Job Opportunities")
        cv_vector = embedder.encode([cv_summary])[0]
        match_scores = []
        for job in jobs:
            job_vector = embedder.encode([job["description"]])[0]
            score = cosine_similarity([cv_vector], [job_vector])[0][0]
            match_scores.append((job, score))

        for i, (job, score) in enumerate(match_scores):
            match_percent = round(score * 100, 2)
            st.markdown(f"### 🔹 [{job['title']} - {job['location']}]({job['url']}) — {match_percent}% Match")
            with st.expander("📝 View Details"):
                st.markdown(job["description"])
                advice = ask_deepseek(f"Evaluate this job match:\nJob: {job['title']}\n{job['description']}\nCV: {cv_summary}")
                st.success(advice)

                tailoring_prompt = f"Write a full tailored CV for this job:\n{job['description']}\nOriginal CV:\n{cv_summary}"
                if st.button("✍️ Generate Tailored CV", key=f"cv_{i}"):
                    tailored_cv = ask_deepseek(tailoring_prompt)
                    st.text_area("📄 Tailored CV", tailored_cv, height=400)
                    pdf_file = generate_pdf(tailored_cv)
                    st.download_button("📥 Download as PDF", data=pdf_file, file_name="Tailored_CV.pdf")
                    st.button("📧 Do you want to email your tailored CV and cover letter?", key=f"email_dummy_{i}")

                if score >= 0.5:
                    try:
                        send_whatsapp_alert(f"📬 Job Match Alert!\n{job['title']} ({match_percent}%)\n{job['url']}")
                        st.success("📲 WhatsApp alert sent!")
                    except Exception as e:
                        st.warning(f"❌ WhatsApp alert failed: {str(e)}")

    st.subheader("📈 CV Quality Score (AI)")
    score_feedback = ask_deepseek(f"Score this CV out of 100 and explain briefly:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{score_feedback}</div>', unsafe_allow_html=True)

    st.subheader("🤖 Ask the AI About Your Career or CV")
    user_q = st.text_input("💬 Type your question:")
    if user_q:
        ai_response = ask_deepseek(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {ai_response}")
