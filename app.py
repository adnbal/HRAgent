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
    st.error("\U0001F510 Missing Twilio credentials.")
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
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"\u274c DeepSeek error: {response.text}")
        return ""

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
st.set_page_config(page_title="\U0001F680 AI CV Matcher", layout="wide")
st.title("\U0001F31F AI-Based CV Matcher with PDF Export, WhatsApp Alerts, and Tailored Applications")

uploaded_file = st.file_uploader("\U0001F4C4 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("\U0001F4C4 Analyzing your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_summary = ask_deepseek(f"Summarize this CV:\n{cv_text}")

        # Extract job roles
        role_prompt = f"""
        From this CV summary, extract the top 5 job roles the candidate is suited for.
        Return as a plain bullet list, no explanations.

        CV Summary:
        {cv_summary}
        """
        role_output = ask_deepseek(role_prompt)
        st.subheader("\U0001F4BC Best Roles Suited for You")
        st.markdown(f'<div class="neon-box">{role_output}</div>', unsafe_allow_html=True)

        # Use first role for job matching
        first_valid = None
        for line in role_output.split("\n"):
            clean = re.sub(r"[^a-zA-Z\s]", "", line).strip().lower()
            if 3 <= len(clean) <= 40:
                first_valid = clean
                break
        if not first_valid:
            first_valid = "data analyst"
            st.warning("⚠️ No valid role extracted. Using default: Data Analyst")

        jobs = fetch_dummy_jobs(first_valid)

        st.subheader("\U0001F4CA Matched Job Opportunities")
        cv_vector = embedder.encode([cv_summary])[0]
        match_scores = []
        for job in jobs:
            job_vector = embedder.encode([job["description"]])[0]
            score = cosine_similarity([cv_vector], [job_vector])[0][0]
            match_scores.append((job, score))

        for i, (job, score) in enumerate(match_scores):
            match_percent = round(score * 100, 2)
            st.markdown(f"### \U0001F539 [{job['title']} - {job['location']}]({job['url']}) — {match_percent}% Match")
            with st.expander("\U0001F4DD View Details"):
                st.markdown(job["description"])
                advice = ask_deepseek(f"Evaluate this job for the CV:\nJob: {job['title']}\n{job['description']}\nCV: {cv_summary}")
                st.success(advice)

                tailoring_prompt = f"Write a tailored CV for this job:\n{job['description']}\nOriginal CV:\n{cv_summary}"
                if st.button("✍️ Generate Tailored CV", key=f"cv_{i}"):
                    tailored_cv = ask_deepseek(tailoring_prompt)
                    st.text_area("\U0001F4C4 Tailored CV", tailored_cv, height=400)
                    pdf_file = generate_pdf(tailored_cv)
                    st.download_button("\U0001F4E5 Download as PDF", data=pdf_file, file_name="Tailored_CV.pdf")
                    st.button("\U0001F4E7 Do you want to email your tailored CV and cover letter?", key=f"email_{i}")

                if score >= 0.5:
                    try:
                        send_whatsapp_alert(f"\U0001F4EC Job Match Alert!\n{job['title']} ({match_percent}%)\n{job['url']}")
                        st.success("\U0001F4F2 WhatsApp alert sent!")
                    except Exception as e:
                        st.warning(f"❌ WhatsApp failed: {str(e)}")

    st.subheader("\U0001F4C8 CV Quality Score (AI)")
    score_prompt = f"Score this CV out of 100 and explain briefly:\n{cv_summary}"
    quality_feedback = ask_deepseek(score_prompt)
    st.markdown(f'<div class="neon-box">{quality_feedback}</div>', unsafe_allow_html=True)

    st.subheader("\U0001F916 Ask the AI About Your Career or CV")
    user_q = st.text_input("\U0001F4AC Type your question:")
    if user_q:
        ai_response = ask_deepseek(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**\U0001F9E0 AI Answer:** {ai_response}")
