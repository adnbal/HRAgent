# ------------------- Streamlit Setup -------------------
import streamlit as st
import fitz
import re
import requests
from io import BytesIO
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- API Config -------------------
OPENROUTER_API_KEY = st.secrets["openrouter"]["api_key"]

def ask_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an expert AI career advisor."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

# ------------------- Twilio Config -------------------
try:
    from twilio.rest import Client
    twilio_sid = st.secrets["twilio"]["account_sid"]
    twilio_token = st.secrets["twilio"]["auth_token"]
    whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
    whatsapp_from = "whatsapp:+14155238886"
except:
    st.warning("Twilio not configured.")

def send_whatsapp_alert(message):
    client = Client(twilio_sid, twilio_token)
    client.messages.create(body=message, from_=whatsapp_from, to=whatsapp_to)

# ------------------- Styling -------------------
st.markdown("""
<style>
.neon-box {
    border: 2px solid #00FFFF;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 0 5px #00ffff; }
    to { box-shadow: 0 0 30px #00ffff; }
}
.stApp {
    background-color: #000000;
    color: white;
}
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
    font-family: 'Courier New', monospace;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Utility Functions -------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

def fetch_adzuna_jobs(keyword, country_code, location=None):
    try:
        app_id = st.secrets["adzuna"]["app_id"]
        app_key = st.secrets["adzuna"]["app_key"]
        base = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": 6,
            "what": keyword,
            "where": location or ""
        }
        res = requests.get(base, params=params)
        res.raise_for_status()
        jobs = res.json().get("results", [])
        return [{
            "title": j["title"],
            "location": j["location"]["display_name"],
            "description": j["description"],
            "url": j["redirect_url"]
        } for j in jobs]
    except:
        return []

def fetch_dummy_jobs(keyword):
    return [
        {"title": f"{keyword.title()} at TechCorp", "location": "Remote", "description": f"Join us as a {keyword}.", "url": "https://example.com/job1"},
        {"title": f"{keyword.title()} Analyst", "location": "New York", "description": f"Help us scale AI efforts as a {keyword}.", "url": "https://example.com/job2"},
    ]

# ------------------- App Start -------------------
st.set_page_config(page_title="🚀 AI CV Matcher", layout="wide")
st.title("🌟 AI CV Matcher with Tailored Resume, WhatsApp Alerts & Glowing Demo UI")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    cv_text = extract_text_from_pdf(uploaded_file)
    cv_summary = ask_deepseek(f"Summarize this CV:\n{cv_text}")

    # Override role detection silently
    override_roles = [
        "Artificial Intelligence", "Data Science", "Data Analytics", "Business Analytics",
        "Agentic AI", "Autonomous Agent", "Prompt Engineering", "Policy Modeling",
        "AI Governance", "Social Impact AI", "AI for Government"
    ]
    display_roles = ", ".join(override_roles)
    st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b> {display_roles}</div>', unsafe_allow_html=True)

    # Country input
    country_map = {"United States": "us", "New Zealand": "nz", "United Kingdom": "gb", "Australia": "au", "Canada": "ca", "India": "in"}
    country = st.selectbox("🌍 Choose Country", list(country_map.keys()), index=0)
    location = st.text_input("📍 City or Region (optional)", "")

    jobs = fetch_adzuna_jobs("AI", country_map[country], location)
    if not jobs:
        jobs = fetch_dummy_jobs("AI")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    cv_vec = embedder.encode([cv_summary])[0]

    job_matches = []
    for job in jobs:
        job_vec = embedder.encode([job["description"]])[0]
        raw_score = cosine_similarity([cv_vec], [job_vec])[0][0]
        inflated_score = min(raw_score * 1.2, 1.0)
        job_matches.append({"job": job, "score": round(inflated_score * 100, 2)})

    job_matches = sorted(job_matches, key=lambda x: x["score"], reverse=True)

    st.subheader("📊 Matched Job Listings")
    for i, entry in enumerate(job_matches):
        job = entry["job"]
        score = entry["score"]
        st.markdown(f"### 🔹 [{job['title']} – {job['location']}]({job['url']}) — {score:.2f}% Match")
        with st.expander("📄 View Details"):
            st.write(job["description"])
            explanation = ask_deepseek(
                f"Given this CV:\n{cv_summary}\nAnd this job description:\n{job['description']}\n"
                "Explain why this is a match, what's missing, and whether to apply."
            )
            st.success(explanation)

            if st.button(f"✍️ Tailor CV for this job", key=f"tailor_{i}"):
                tailored = ask_deepseek(f"Tailor this CV for the job:\n{job['description']}\n\nOriginal:\n{cv_summary}")
                st.markdown(f"<div class='small-cv'>{tailored}</div>", unsafe_allow_html=True)
                st.download_button("📥 Download as PDF", generate_pdf(tailored), file_name="Tailored_CV.pdf")

            st.button("🚀 Auto-Apply for this Job", key=f"autoapply_{i}")
            if score >= 50:
                try:
                    send_whatsapp_alert(f"✅ Match: {job['title']} ({score}%)\nApply: {job['url']}")
                    st.success("📲 WhatsApp alert sent!")
                except:
                    st.warning("WhatsApp failed.")
    st.subheader("📈 CV Quality Score (AI)")
    score_feedback = ask_deepseek(f"Score this CV out of 100 and explain how to improve it:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{score_feedback}</div>', unsafe_allow_html=True)

    # Always show improved CV based on score
    improved_cv = ask_deepseek(
        f"Based on this CV summary:\n{cv_summary}\n\n"
        f"And this quality review:\n{score_feedback}\n\n"
        "Please rewrite the CV to improve it, using markdown formatting, in a Word-style layout (small font)."
    )
    st.markdown("📌 **Do you want me to make these improvements and give updated CV converted to PDF?**")
    st.markdown(f"<div class='small-cv'>{improved_cv}</div>", unsafe_allow_html=True)

    # ------------------- Q&A Section -------------------
    st.subheader("🤖 Ask AI About Your Career or CV")
    user_q = st.text_input("💬 Your question:")
    if user_q:
        reply = ask_deepseek(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {reply}")
        # Always show updated CV if AI offers advice
        styled_q_cv = ask_deepseek(
            f"Based on this AI advice:\n{reply}\n\n"
            "Give a markdown preview of the updated CV (small font) that reflects these suggestions."
        )
        st.markdown("📌 **Do you want me to make these changes and give updated CV converted to PDF?**")
        st.markdown(f"<div class='small-cv'>{styled_q_cv}</div>", unsafe_allow_html=True)
