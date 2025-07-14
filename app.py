import streamlit as st
import fitz
import re
import requests
from fpdf import FPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ------------------- OpenAI CONFIG -------------------
openai.api_key = st.secrets["openai"]["api_key"]
if "organization" in st.secrets["openai"]:
    openai.organization = st.secrets["openai"]["organization"]

def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert AI career advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ OpenAI API error: {e}")
        return "Sorry, I couldn't generate a response due to an API error."

# ------------------- Twilio WhatsApp -------------------
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

# ------------------- CSS Styling -------------------
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

# ------------------- Utilities -------------------
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
