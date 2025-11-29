Resume Screening Agent — AI-Powered Resume Ranking System

This project is an AI-powered Resume Screening Agent that analyzes job descriptions and ranks uploaded resumes using semantic similarity. It uses vector embeddings, a local vector store, cosine similarity ranking, and LLM-based explanations for the top candidates. The app runs on Streamlit and processes PDF/DOCX resumes automatically.

🚀 Features
Core Functionality

Upload multiple resumes (PDF/DOCX)

Enter or upload a Job Description (JD)

Automatically extracts text using PDF/DOCX parsers

Generates text embeddings using OpenAI (or fallback local embeddings)

Stores vectors in a local vector store (ChromaDB-compatible structure)

Computes similarity using cosine similarity

Shows ranked candidates with similarity score

Provides LLM explanations for top resumes

Export results to CSV

🧱 Architecture Overview

The system consists of:

Frontend (Streamlit UI)

Upload JD

Upload resumes

Display ranking result

Select model for explanations

Parser Layer

PDF text extraction

DOCX extraction

Normalization & cleanup

Embedding Layer

OpenAI embeddings (preferred)

Automatic fallback to deterministic local embeddings when quota exceeded

Vector Store

Stores embeddings + metadata

Efficient top-k semantic search

Ranker

Converts embeddings → cosine similarity

Returns sorted ranking

Explanation Agent

Uses LLM (GPT-4o-mini or fallback)

Generates short “Why this candidate?” explanation

🛠️ Tech Stack
Programming

Python 3.10+

Streamlit UI

Libraries

numpy

python-docx

PyPDF2

Chroma (or custom in-memory store wrapper)

OpenAI Python SDK

Pandas

APIs Used

OpenAI Embeddings

OpenAI LLMs for explanations

🧩 Project Structure
resume-screening-agent-updated/
│── app.py
│── requirements.txt
│── README.md
│── sample_data/
│── resumes/
│── src/
│    ├── parser.py
│    ├── ranker.py
│    ├── embedder.py
│    ├── vector_store.py
│    ├── explain.py
│    ├── resume.py
│    └── __init__.py
└── venv/

⚙️ Setup & Run Instructions
1. Clone the Repository
git clone https://github.com/pavani095/resume-screening-agent-updated.git
cd resume-screening-agent-updated

2. Create Virtual Environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate

3. Install Libraries
pip install -r requirements.txt

4. Set Your API Key

PowerShell

$env:OPENAI_API_KEY="your_api_key_here"

5. Run the App
streamlit run app.py
<img width="1024" height="1536" alt="architecture" src="https://github.com/user-attachments/assets/ad295071-f988-4fdf-991e-baf8e515c7e3" />


Your local app will open automatically at:

http://localhost:8501
