import streamlit as st
import fitz  # PyMuPDF
import re
import requests
from fpdf import FPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ------------------- CONFIG -------------------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
try:
    twilio_sid = st.secrets["twilio"]["account_sid"]
    twilio_token = st.secrets["twilio"]["auth_token"]
    whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
    whatsapp_from = "whatsapp:+14155238886"
except KeyError:
    st.error("❌ Missing Twilio or OpenAI credentials.")
    st.stop()

def send_whatsapp_alert(message):
    from twilio.rest import Client
    client = Client(twilio_sid, twilio_token)
    client.messages.create(body=message, from_=whatsapp_from, to=whatsapp_to)

# ------------------- STYLE -------------------
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
</style>
""", unsafe_allow_html=True)

# ------------------- UTILS -------------------
def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

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

# ------------------- APP UI -------------------
st.set_page_config(page_title="🚀 AI CV Matcher", layout="wide")
st.title("🌟 AI CV Matcher with Tailored Resume, WhatsApp Alerts & Glowing Demo UI")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Reading your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_summary = ask_openai(f"Summarize this CV:\n{cv_text}")

    keyword_prompt = """
    From this CV summary, extract the top 3 job roles best suited for the candidate.
    Output just the roles in plain list like:
    - Data Analyst
    - AI Engineer
    - Business Consultant
    """
    raw_keywords = ask_openai(f"{keyword_prompt}\nCV Summary:\n{cv_summary}")
    roles = [re.sub(r"[-•0-9]", "", r).strip() for r in raw_keywords.strip().split("\n")]
    valid_roles = [r for r in roles if 3 <= len(r) <= 40 and " " in r]
    search_keyword = valid_roles[0].lower() if valid_roles else "data analyst"

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

    st.subheader("📊 Matched Job Listings")
    for i, job in enumerate(jobs):
        job_vec = embedder.encode([job["description"]])[0]
        match = cosine_similarity([cv_vec], [job_vec])[0][0]
        match_pct = round(match * 100, 2)

        st.markdown(f"### 🔹 [{job['title']} – {job['location']}]({job['url']}) — {match_pct}% Match")
        with st.expander("📄 View Details"):
            st.write(job["description"])
            reasoning = ask_openai(
                f"Given this CV summary:\n{cv_summary}\n\nAnd this job:\n{job['description']}\n\n"
                "Why is this a good match? What's missing? Should the candidate apply?"
            )
            st.success(reasoning)

            if st.button(f"✍️ Tailor CV for this job", key=f"tailor_{i}"):
                tailored_cv = ask_openai(
                    f"Write a tailored version of this CV for the job below.\nJob: {job['description']}\nOriginal CV:\n{cv_summary}"
                )
                st.text_area("📄 Tailored CV", tailored_cv, height=400)

                pdf = generate_pdf(tailored_cv)
                st.download_button("📥 Download as PDF", pdf, file_name="Tailored_CV.pdf")

                st.button("📧 Do you want to email your tailored CV and cover letter?", key=f"email_{i}")

            st.button("🤖 Auto-Apply for this Job (Dummy)", key=f"autoapply_{i}")

            if match >= 0.5:
                try:
                    send_whatsapp_alert(f"✅ Match: {job['title']} ({match_pct}%)\nApply: {job['url']}")
                    st.success("📲 WhatsApp alert sent!")
                except Exception as e:
                    st.warning(f"❌ WhatsApp failed: {e}")

    st.subheader("📈 CV Quality Score (AI)")
    score_response = ask_openai(f"Score this CV out of 100 and explain briefly:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{score_response}</div>', unsafe_allow_html=True)

    st.subheader("🤖 Ask AI About Your Career or CV")
    user_q = st.text_input("💬 Your question:")
    if user_q:
        reply = ask_openai(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {reply}")

        if "You should" in reply or "consider" in reply:
            styled_preview = ask_openai(
                f"Based on these suggestions:\n{reply}\n\n"
                "Give a styled preview of the updated CV in markdown format (visually appealing)."
            )
            st.markdown("📌 **Do you want me to make these changes and give updated CV converted to PDF?**")
            st.markdown(styled_preview, unsafe_allow_html=True)
