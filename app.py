import json
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

KB_FILE = "knowledge_base.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

with open(KB_FILE, "r", encoding="utf-8") as f:
    passages = json.load(f)

texts = [p["text"] for p in passages]
ids = [p["id"] for p in passages]

model = SentenceTransformer(MODEL_NAME)
passage_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

def clean_question(question: str) -> str:
    # Remove obvious prompt-injection instructions.
    lowered = question.lower()
    injection_patterns = [
        r"ignore (all )?(previous|prior) instructions",
        r"disregard .{0,30}instructions",
        r"reveal .{0,30}(system|instructions|prompt)",
        r"you are now",
        r"act as",
        r"system prompt",
        r"developer message"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, lowered):
            return ""
    return question.strip()

def abstain_response():
    return {
        "answer": None,
        "sources": [],
        "scores": [],
        "confidence": 0.0
    }

@app.post("/query")
def query(request: QueryRequest):
    question = clean_question(request.question)

    if not question:
        return abstain_response()

    top_k = max(1, min(request.top_k or 3, 10))

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    scores = passage_embeddings @ question_embedding
    ranked = np.argsort(scores)[::-1][:top_k]

    sources = [int(ids[i]) for i in ranked]
    passage_scores = [float(scores[i]) for i in ranked]

    best_score = passage_scores[0]
    second_score = passage_scores[1] if len(passage_scores) > 1 else 0.0

    # Conservative retrieval threshold.
    if best_score < 0.42:
        return abstain_response()

    # Ambiguous query: two unrelated-looking passages score similarly.
    margin = best_score - second_score
    if best_score < 0.50 and margin < 0.03:
        return abstain_response()

    confidence = min(0.99, max(0.0, (best_score - 0.35) / 0.55))

    answer = extract_answer(question, texts[ranked[0]])

    if not answer:
        return abstain_response()

    return {
        "answer": answer,
        "sources": sources,
        "scores": passage_scores,
        "confidence": confidence
    }

def extract_answer(question: str, passage: str) -> str | None:
    q = question.lower()

    # Return the whole passage when no reliable extraction rule matches.
    # This is safer than inventing an answer.
    return passage
