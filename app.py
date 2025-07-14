import streamlit as st
import fitz
import re
import requests
from fpdf import FPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- CONFIG -------------------
OPENROUTER_API_KEY = st.secrets["openrouter"]["api_key"]

def ask_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com/",
        "X-Title": "cv-analyzer",
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

try:
    twilio_sid = st.secrets["twilio"]["account_sid"]
    twilio_token = st.secrets["twilio"]["auth_token"]
    whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
    whatsapp_from = "whatsapp:+14155238886"
except KeyError:
    st.error("❌ Missing Twilio or OpenRouter credentials.")
    st.stop()

def send_whatsapp_alert(message):
    from twilio.rest import Client
    client = Client(twilio_sid, twilio_token)
    client.messages.create(body=message, from_=whatsapp_from, to=whatsapp_to)

# ------------------- STYLING -------------------
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
    white-space: pre-wrap;
    font-family: 'Courier New', monospace;
}
</style>
""", unsafe_allow_html=True)

# ------------------- UTILS -------------------
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
    except Exception:
        return []

def fetch_dummy_jobs(keyword):
    return [
        {"title": f"{keyword.title()} at TechCorp", "location": "Remote", "description": f"Join us as a {keyword}.", "url": "https://example.com/job1"},
        {"title": f"{keyword.title()} Specialist", "location": "New York", "description": f"We're looking for a {keyword} expert.", "url": "https://example.com/job2"},
        {"title": f"Lead {keyword.title()}", "location": "London", "description": f"Lead our {keyword} division.", "url": "https://example.com/job3"},
    ]

# ------------------- MAIN APP -------------------
st.set_page_config(page_title="🚀 AI CV Matcher", layout="wide")
st.title("🌟 AI CV Matcher with Tailored Resume, WhatsApp Alerts & Glowing Demo UI")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Reading your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_summary = ask_deepseek(f"Summarize this CV:\n{cv_text}")

    keyword_prompt = """
    From this CV summary, extract the top 3 job roles most relevant to the candidate.
    Prefer roles in artificial intelligence, data science, business analytics, financial analytics, banking analytics, and policy-making.
    Format:
    - Data Scientist
    - AI Consultant
    - Policy Advisor
    """
    raw_keywords = ask_deepseek(f"{keyword_prompt}\nCV Summary:\n{cv_summary}")
    roles = [re.sub(r"[-•0-9]", "", r).strip() for r in raw_keywords.strip().split("\n")]
    valid_roles = [r for r in roles if 3 <= len(r) <= 40]
    search_keyword = valid_roles[0].lower() if valid_roles else "data scientist"

    st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b> {search_keyword.title()}</div>', unsafe_allow_html=True)

    country_map = {
        "United States": "us", "New Zealand": "nz", "United Kingdom": "gb",
        "Australia": "au", "Canada": "ca", "India": "in"
    }
    country = st.selectbox("🌍 Choose Country", list(country_map.keys()), index=0)
    location = st.text_input("📍 City or Region (optional)", "")

    jobs = fetch_adzuna_jobs(search_keyword, country_map[country], location)
    if not jobs:
        st.warning("⚠️ No live jobs found, showing fallback examples.")
        jobs = fetch_dummy_jobs(search_keyword)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    cv_vec = embedder.encode([cv_summary])[0]

    job_matches = []
    for job in jobs:
        job_vec = embedder.encode([job["description"]])[0]
        raw_match = cosine_similarity([cv_vec], [job_vec])[0][0]
        inflated_match = min(raw_match * 1.2, 1.0)
        job_matches.append({"job": job, "match_pct": round(inflated_match * 100, 2)})

    job_matches = sorted(job_matches, key=lambda x: x["match_pct"], reverse=True)

    st.subheader("📊 Matched Job Listings")
    for i, entry in enumerate(job_matches):
        job = entry["job"]
        match = entry["match_pct"]

        st.markdown(f"### 🔹 [{job['title']} – {job['location']}]({job['url']}) — {match:.2f}% Match")
        with st.expander("📄 View Details"):
            st.write(job["description"])
            reasoning = ask_deepseek(
                f"Given this CV summary:\n{cv_summary}\n\nAnd this job:\n{job['description']}\n\n"
                "Why is this a good match? What's missing? Should the candidate apply?"
            )
            st.success(reasoning)

            if st.button(f"✍️ Tailor CV for this job", key=f"tailor_{i}"):
                tailored_cv = ask_deepseek(
                    f"Write a tailored version of this CV for the job below.\nJob: {job['description']}\nOriginal CV:\n{cv_summary}"
                )
                st.markdown(f"<div class='small-cv'>{tailored_cv}</div>", unsafe_allow_html=True)
                pdf = generate_pdf(tailored_cv)
                st.download_button("📥 Download as PDF", pdf, file_name="Tailored_CV.pdf")
                st.button("📧 Do you want to email your tailored CV and cover letter?", key=f"email_{i}")

            st.button("🚀 Auto-Apply for this Job", key=f"autoapply_{i}")

            if match >= 50:
                try:
                    send_whatsapp_alert(f"✅ Match: {job['title']} ({match:.2f}%)\nApply: {job['url']}")
                    st.success("📲 WhatsApp alert sent!")
                except Exception as e:
                    st.warning(f"❌ WhatsApp failed: {e}")

    st.subheader("📈 CV Quality Score (AI)")
    score_response = ask_deepseek(f"Score this CV out of 100 and explain briefly:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{score_response}</div>', unsafe_allow_html=True)

    st.subheader("🤖 Ask AI About Your Career or CV")
    user_q = st.text_input("💬 Your question:")
    if user_q:
        reply = ask_deepseek(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {reply}")

        styled_preview = ask_deepseek(
            f"Based on these suggestions:\n{reply}\n\n"
            "Give a styled preview of the updated CV in markdown format (small font)."
        )
        st.markdown("📌 **Do you want me to make these changes and give updated CV converted to PDF?**")
        st.markdown(f"<div class='small-cv'>{styled_preview}</div>", unsafe_allow_html=True)
