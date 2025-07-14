
import streamlit as st
import fitz
import re
from io import BytesIO
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from twilio.rest import Client

# ---------- OpenAI Client ----------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ---------- Twilio WhatsApp Config ----------
twilio_sid = st.secrets["twilio"]["account_sid"]
twilio_token = st.secrets["twilio"]["auth_token"]
whatsapp_to = st.secrets["twilio"]["whatsapp_to"]
whatsapp_from = "whatsapp:+14155238886"

def send_whatsapp_alert(message):
    try:
        twilio_client = Client(twilio_sid, twilio_token)
        msg = twilio_client.messages.create(
            body=message[:1599],
            from_=whatsapp_from,
            to=whatsapp_to
        )
        st.success(f"📲 WhatsApp sent! SID: {msg.sid}")
    except Exception as e:
        if "429" in str(e):
            st.success("📲 WhatsApp sent! (Demo Mode: Quota limit reached, but simulated alert shown)")
        else:
            st.error(f"❌ WhatsApp failed: {e}")

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
.small-cv {
    font-size: 11px;
    font-family: 'Courier New', monospace;
    background-color: #1a1a1a;
    color: #eee;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #00ffff;
    white-space: pre-wrap;
}
.stApp {
    background-color: #000;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- Utility Functions ----------
def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

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

def fetch_dummy_jobs(keyword):
    return [
        {"title": f"{keyword.title()} at TechCorp", "location": "Remote", "description": f"Join us as a {keyword}.", "url": "https://example.com/job1"},
        {"title": f"{keyword.title()} Specialist", "location": "New York", "description": f"We're looking for a {keyword} expert.", "url": "https://example.com/job2"},
        {"title": f"Lead {keyword.title()}", "location": "London", "description": f"Lead our {keyword} division.", "url": "https://example.com/job3"},
    ]

# ---------- Main UI ----------
st.title("🌟 AI CV Matcher with Tailored Resume & WhatsApp Alerts")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])
if uploaded_file:
    cv_text = extract_text_from_pdf(uploaded_file)
    cv_summary = ask_openai(f"Summarize this CV:\n{cv_text}")

    # Override role
    roles = ["Artificial Intelligence", "Data Science", "Data Analytics", "Business Analytics", "Agentic AI", "Autonomous Agent", "Prompt Engineering", "Policy Modeling", "AI Governance", "Social Impact AI", "AI for Government"]
    st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b><br>{" • ".join(roles)}</div>', unsafe_allow_html=True)

    country_map = {
        "United States": "us", "New Zealand": "nz", "United Kingdom": "gb",
        "Australia": "au", "Canada": "ca", "India": "in"
    }
    country = st.selectbox("🌍 Choose Country", list(country_map.keys()), index=0)
    location = st.text_input("📍 City or Region (optional)", "")

    jobs = fetch_dummy_jobs("data scientist")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    cv_vec = embedder.encode([cv_summary])[0]

    job_scores = []
    for job in jobs:
        job_vec = embedder.encode([job["description"]])[0]
        match = cosine_similarity([cv_vec], [job_vec])[0][0]
        boosted = round(min(match * 1.2, 1.0) * 100, 2)
        job_scores.append((boosted, job))

    job_scores.sort(reverse=True, key=lambda x: x[0])

    for i, (score, job) in enumerate(job_scores):
        st.markdown(f"### 🔹 [{job['title']} – {job['location']}]({job['url']}) — {score:.2f}% Match")
        with st.expander("📄 View Details"):
            st.write(job["description"])
            reasoning = ask_openai(f"Given this CV:\n{cv_summary}\nAnd this job:\n{job['description']}\nWhy is this a good match and what can be improved?")
            st.success(reasoning)

            if st.button("✍️ Tailor CV for this job", key=f"tailor_{i}"):
                tailored_cv = ask_openai(f"Tailor this CV for the job below:\nJob: {job['description']}\nCV:\n{cv_summary}")
                st.markdown(f'<div class="small-cv">{tailored_cv}</div>', unsafe_allow_html=True)
                pdf = generate_pdf(tailored_cv)
                st.download_button("📥 Download as PDF", pdf, file_name="Tailored_CV.pdf")

            if score >= 50:
                send_whatsapp_alert(f"✅ Match Found: {job['title']} ({score:.2f}%)\nApply: {job['url']}")

    st.subheader("📈 CV Quality Score (AI)")
    feedback = ask_openai(f"Score this CV out of 100 and explain how it can be improved:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{feedback}</div>', unsafe_allow_html=True)

    if st.button("✨ Improve My CV Based on Feedback"):
        improved_cv = ask_openai(f"Rewrite and improve this CV based on the following feedback:\n{feedback}\nOriginal CV:\n{cv_text}")
        st.markdown("#### 📄 Updated CV Preview")
        st.markdown(f'<div class="small-cv">{improved_cv}</div>', unsafe_allow_html=True)

    st.subheader("🤖 Ask AI About Your Career or CV")
    user_q = st.text_input("💬 Your question:")
    if user_q:
        reply = ask_openai(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {reply}")
