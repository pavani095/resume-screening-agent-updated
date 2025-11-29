# src/explain.py
import os, traceback, re
USE_OPENAI = bool(os.environ.get('OPENAI_API_KEY'))
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) if USE_OPENAI else None
except Exception:
    client = None
    USE_OPENAI = False

def _heuristic_explain(jd_text: str, resume_text: str) -> str:
    jd_words = set(w.lower() for w in re.findall(r"\w+", jd_text) if len(w)>2)
    resume_words = set(w.lower() for w in re.findall(r"\w+", resume_text) if len(w)>2)
    common = jd_words.intersection(resume_words)
    m = re.search(r"(\d{1,2})\+?\s+years", resume_text.lower())
    years = f"{m.group(1)} years" if m else "years not specified"
    top_skills = ", ".join(list(common)[:6]) or "no exact keyword matches"
    return f"{years} experience; matched keywords: {top_skills}."

DEFAULT_MODEL = "gpt-4o-mini"

def explain_candidate(jd_text: str, resume_text: str, model: str = None) -> str:
    if not jd_text or not resume_text:
        return "Insufficient information to explain."
    if model is None:
        model = DEFAULT_MODEL
    if USE_OPENAI and client is not None:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":f"Given the job description:\n{jd_text}\n\nCandidate resume:\n{resume_text}\n\nIn 2-3 lines, explain why candidate is a good or poor match."}],
                max_tokens=200,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI explain failed, falling back to heuristic.", e)
            traceback.print_exc()
            return _heuristic_explain(jd_text, resume_text)
    else:
        return _heuristic_explain(jd_text, resume_text)
