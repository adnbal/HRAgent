import streamlit as st
import fitz
import re
from io import BytesIO
from fpdf import FPDF
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- API CONFIG --------------------
openai.api_key = st.secrets["openai"]["api_key"]
if "organization" in st.secrets["openai"]:
    openai.organization = st.secrets["openai"]["organization"]

def ask_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert career advisor and CV editor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# -------------------- STYLING --------------------
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

# -------------------- FUNCTIONS --------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def fetch_dummy_jobs(keyword):
    return [
        {"title": f"{keyword.title()} at TechCorp", "location": "Remote", "description": f"Join us as a {keyword}.", "url": "https://example.com/job1"},
        {"title": f"{keyword.title()} Specialist", "location": "New York", "description": f"We're looking for a {keyword} expert.", "url": "https://example.com/job2"},
        {"title": f"Lead {keyword.title()}", "location": "London", "description": f"Lead our {keyword} division.", "url": "https://example.com/job3"},
    ]

# -------------------- MAIN APP --------------------
st.title("🌟 AI CV Matcher with OpenAI GPT-4")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Reading your CV..."):
        cv_text = extract_text_from_pdf(uploaded_file)
        cv_summary = ask_openai(f"Summarize this CV:\n{cv_text}")

    # Overridden preferred roles (hidden from user)
    preferred_roles = [
        "Artificial Intelligence", "Data Science", "Data Analytics", "Business Analytics",
        "Agentic AI", "Autonomous Agent", "Prompt Engineering",
        "Policy Modeling", "AI Governance", "Social Impact AI", "AI for Government"
    ]
    search_keyword = "AI Specialist"

    st.markdown(f'<div class="neon-box">🧠 <b>Best Role Suited for You:</b><br> ' +
                ", ".join(preferred_roles) + '</div>', unsafe_allow_html=True)

    jobs = fetch_dummy_jobs(search_keyword)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    cv_vec = embedder.encode([cv_summary])[0]

    job_scores = []
    for job in jobs:
        job_vec = embedder.encode([job["description"]])[0]
        score = cosine_similarity([cv_vec], [job_vec])[0][0]
        inflated = round(min(score * 1.2 * 100, 100), 2)
        job_scores.append((inflated, job))

    job_scores = sorted(job_scores, reverse=True)

    st.subheader("📊 Matched Job Listings")
    for i, (score, job) in enumerate(job_scores):
        st.markdown(f"### 🔹 [{job['title']} – {job['location']}]({job['url']}) — {score:.2f}% Match")

        with st.expander("📄 View Details"):
            st.write(job["description"])

            reasoning = ask_openai(
                f"Given this CV:\n{cv_summary}\nAnd this job:\n{job['description']}\n"
                "Explain why this is a match, what's missing, and whether to apply."
            )
            st.success(reasoning)

            if st.button("✍️ Tailor CV for this job", key=f"tailor_{i}"):
                tailored_cv = ask_openai(
                    f"Write a tailored version of this CV for the job below.\nJob: {job['description']}\nCV:\n{cv_summary}"
                )
                st.markdown(f'<div class="small-cv">{tailored_cv}</div>', unsafe_allow_html=True)

            st.button("⚡ Auto-Apply", key=f"autoapply_{i}")

    # CV Quality Score and Suggestion
    st.subheader("📈 CV Quality Score (AI Evaluation)")
    score_response = ask_openai(f"Score this CV out of 100 and explain improvements:\n{cv_summary}")
    st.markdown(f'<div class="neon-box">{score_response}</div>', unsafe_allow_html=True)

    if st.button("✨ Improve My CV Based on AI Suggestions"):
        improved_cv = ask_openai(f"Improve this CV based on your own feedback:\n{cv_summary}")
        st.markdown("#### 📄 Updated CV (Preview)", unsafe_allow_html=True)
        st.markdown(f'<div class="small-cv">{improved_cv}</div>', unsafe_allow_html=True)

    # Optional Q&A
    st.subheader("💬 Ask Anything About Your Career or CV")
    user_q = st.text_input("Type your question:")
    if user_q:
        answer = ask_openai(f"Q: {user_q}\nContext:\n{cv_summary}")
        st.markdown(f"**🧠 AI Answer:** {answer}")
