import streamlit as st
import fitz  # PyMuPDF
import re
from io import BytesIO
from fpdf import FPDF
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from twilio.rest import Client

# ---------- OpenAI GPT-4 Setup ----------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert AI career and CV advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# ---------- Twilio WhatsApp Alert ----------
try:
    twilio_sid = st.secrets["twilio"]["account_sid"]
    twilio_token = st.secrets["twilio"]["auth_token"]
    whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
    whatsapp_from = "whatsapp:+14155238886"
    def send_whatsapp_alert(msg):
        Client(twilio_sid, twilio_token).messages.create(body=msg, from_=whatsapp_from, to=whatsapp_to)
except KeyError:
    def send_whatsapp_alert(msg): pass  # Do nothing if Twilio creds missing

# ---------- Styling ----------
st.set_page_config(page_title="🌟 AI CV Matcher", layout="wide")
st.markdown("""
<style>
.neon-box {
    border: 2px solid #00FFFF;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 0 5px #00ffff; }
    to { box-shadow: 0 0 30px #00ffff; }
}
.stApp { background-color: #000000; color: white; }
button {
    border: 1px solid #00FFFF !important;
    color: white !important;
    background-color: black !important;
    border-radius: 8px !important;
    box-shadow: 0 0 8px #00FFFF, 0 0 16px #00FFFF !important;
}
div.stButton > button:hover {
    background-color: #00ffff !important;
    color: black !important;
}
.small-cv {
    font-size: 13px !important;
    line-height: 1.4;
    white-space: pre-wrap;
    font-family: 'Courier New', monospace;
}
</style>
""", unsafe_allow_html=True)

# ---------- Utility Functions ----------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def fetch_dummy_jobs(keyword):
    return [
        {"title": f"{keyword} at AI Labs", "location": "Remote", "description": f"Join our {keyword} team.", "url": "https://example.com/job1"},
        {"title": f"{keyword} Specialist", "location": "New York", "description": f"We're seeking a {keyword} professional.", "url": "https://example.com/job2"},
        {"title": f"Lead {keyword}", "location": "London", "description": f"Lead our {keyword} innovation group.", "url": "https://example.com/job3"},
    ]

# ---------- App Interface ----------
st.title("🌟 AI CV Matcher with GPT-4 & Neon Magic")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Reading your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_summary = ask_openai(f"Summarize this CV:\n{cv_text}")

    # 🚀 Hardcoded preferred roles
    preferred_roles = [
        "Artificial Intelligence", "Data Science", "Data Analytics", "Business Analytics",
        "Agentic AI", "Autonomous Agent", "Prompt Engineering",
        "Policy Modeling", "AI Governance", "Social Impact AI", "AI for Government"
    ]
    st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b><br> ' +
                ", ".join(preferred_roles) + '</div>', unsafe_allow_html=True)

    # 🌍 Country & Location Inputs
    country_map = {
        "United States": "us", "New Zealand": "nz", "United Kingdom": "gb",
        "Australia": "au", "Canada": "ca", "India": "in"
    }
    country = st.selectbox("🌍 Choose Country", list(country_map.keys()), index=0)
    location = st.text_input("📍 City or Region (optional)", "")

    # 🔎 Job Fetching & Matching
    search_keyword = "AI Specialist"
    jobs = fetch_dummy_jobs(search_keyword)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    cv_vec = embedder.encode([cv_summary])[0]

    job_scores = []
    for job in jobs:
        job_vec = embedder.encode([job["description"]])[0]
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        score_boosted = round(min(score * 1.2 * 100, 100), 2)
        job_scores.append((score_boosted, job))

    job_scores = sorted(job_scores, reverse=True)

    # 📊 Show Matches
    st.subheader("📊 Matched Jobs (Sorted & Boosted)")
    for i, (score, job) in enumerate(job_scores):
        st.markdown(f"### 🔹 [{job['title']} – {job['location']}]({job['url']}) — {score:.2f}% Match")
        with st.expander("📄 View Details"):
            st.write(job["description"])

            explanation = ask_openai(
                f"Given this CV:\n{cv_summary}\nJob description:\n{job['description']}\n"
                "Explain the match, gaps, and whether to apply."
            )
            st.success(explanation)

            if st.button("✍️ Tailor CV for this job", key=f"tailor_{i}"):
                tailored_cv = ask_openai(
                    f"Write a full tailored version of this CV for the job below.\nJob: {job['description']}\nCV:\n{cv_summary}"
                )
                st.markdown(f'<div class="small-cv">{tailored_cv}</div>', unsafe_allow_html=True)

            st.button("⚡ Auto-Apply", key=f"autoapply_{i}")

            if score >= 50:
                try:
                    send_whatsapp_alert(f"✅ Match: {job['title']} ({score:.2f}%)\nApply: {job['url']}")
                    st.success("📲 WhatsApp alert sent!")
                except Exception as e:
                    st.warning(f"❌ WhatsApp failed: {e}")

    # 📈 CV Quality & Full Rewrite
    st.subheader("📈 CV Quality Score (AI Review)")
    feedback = ask_openai(f"Evaluate this CV (out of 100) and give improvement suggestions:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{feedback}</div>', unsafe_allow_html=True)

    if st.button("✨ Improve My CV Based on Feedback"):
        improved_cv = ask_openai(f"Rewrite this full CV to reflect all improvements:\n{cv_text}")
        st.markdown("#### 📄 Updated CV Preview", unsafe_allow_html=True)
        st.markdown(f'<div class="small-cv">{improved_cv}</div>', unsafe_allow_html=True)

    # 💬 Q&A Section
    st.subheader("🤖 Ask AI About Your Career or CV")
    user_q = st.text_input("💬 Your question:")
    if user_q:
        reply = ask_openai(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {reply}")
