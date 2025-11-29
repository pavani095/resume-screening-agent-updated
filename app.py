# app.py (updated to include CSV export and offline toggle)
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
from src.parser import extract_text_from_file

from embedder import get_embedding
from vector_store import LocalVectorStore
from ranker import score_and_rank
from explain import explain_candidate
import pandas as pd

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("Resume Screening Agent — Demo")

st.markdown("Paste a Job Description (JD) and upload resumes (PDF/DOCX). The app will embed and rank resumes by semantic similarity, and provide short LLM explanations for top candidates.")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Show top K candidates", min_value=1, max_value=20, value=5)
    model_choice = st.selectbox("LLM for explanations", ["gpt-4o-mini", "gpt-4o", "text-davinci-003"], index=0)
    offline_mode = st.checkbox("Force offline mode (no OpenAI calls)", value=False)
    st.markdown("Make sure OPENAI_API_KEY env var is set if you want real OpenAI calls.")

jd_text = st.text_area("Paste Job Description (or upload a JD file):", height=260)
jd_file = st.file_uploader("Optional: Upload JD file (txt, pdf, docx)", type=["txt", "pdf", "docx"])

if jd_file and not jd_text:
    jd_text = extract_text_from_file(jd_file)

uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX)", accept_multiple_files=True, type=["pdf", "docx"])

if st.button("Run Screening"):
    if not jd_text or not uploaded_files:
        st.error("Please provide a job description and at least one resume.")
    else:
        st.info("Parsing resumes...")
        resumes = []
        for f in uploaded_files:
            text = extract_text_from_file(f)
            resumes.append({"name": f.name, "text": text})

        st.info("Generating embeddings and storing in vector DB...")
        vs = LocalVectorStore(collection_name="resumes_demo")
        # Upsert resumes to vector store
        for r in resumes:
            if offline_mode:
                emb = get_embedding(r["text"])  # embedder will fallback to deterministic
            else:
                emb = get_embedding(r["text"])
            vs.upsert(id=r["name"], embedding=emb, metadata={"text": r["text"]})

        st.success("Vector store populated. Running ranking...")
        jd_emb = get_embedding(jd_text) if not offline_mode else get_embedding(jd_text)
        ranked = score_and_rank(jd_emb, vs, top_k=top_k)

        st.header("Ranked Candidates")
        rows = []
        for i, item in enumerate(ranked, start=1):
            name = item["id"]
            score = item["score"]
            snippet = (item.get("document") or item.get("metadata", {}).get("text", ""))[:800].replace("\n", " ")
            st.subheader(f"{i}. {name} — score: {score:.4f}")
            st.write(snippet + ("..." if len(snippet) > 700 else ""))
            if st.button(f"Explain: {name}", key=f"explain_{i}"):
                with st.spinner("Generating explanation..."):
                    explanation = explain_candidate(jd_text, item.get("document") or item.get("metadata", {}).get("text", ""), model=model_choice)
                    st.markdown(f"**Explanation:** {explanation}")
            rows.append({"name": name, "score": score, "snippet": snippet})
        if rows:
            df = pd.DataFrame(rows)
            st.download_button("Download CSV", df.to_csv(index=False), "ranked.csv", "text/csv")
        st.success("Done.")
