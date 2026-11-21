import json
import re

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

KB_FILE = "knowledge_base.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with open(KB_FILE, "r", encoding="utf-8") as file:
    passages = json.load(file)

passage_ids = [int(p["id"]) for p in passages]
passage_texts = [p["text"] for p in passages]

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
passage_embeddings = embedding_model.encode(
    passage_texts,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

app = FastAPI(title="RAG Colosseum")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


STOPWORDS = {
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "the", "a", "an", "is", "are", "was", "were", "does", "do", "did",
    "of", "in", "on", "for", "to", "and", "or", "with", "by", "from",
    "used", "use", "uses", "using", "require", "requires", "required",
    "provide", "provides", "provided", "according", "based", "following",
    "protocol", "system", "systems", "database", "databases", "technology",
    "please", "tell", "me", "about"
}

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard .{0,40}instructions",
    r"reveal .{0,40}(system|instructions|prompt)",
    r"you are now",
    r"act as",
    r"system prompt",
    r"developer message",
    r"jailbreak",
]

FACTUAL_PATTERNS = [
    r"\bquorum of \d+ (?:nodes|acceptors|members|replicas)\b",
    r"\b\d+ nodes\b",
    r"\b\d+ acceptors\b",
    r"\b\d+ members\b",
    r"\bport \d+\b",
    r"\b\d+ms\b",
    r"\b\d+ seconds\b",
    r"\b\d+ bits\b",
    r"\b\d+ samples\b",
    r"\b\d+ records\b",
    r"\b\d+ bytes\b",
    r"\b\d+ tokens\b",
    r"\b\d+ hidden dense layers\b",
    r"\b\d+ transformer blocks\b",
    r"\b\d+ residual bottleneck blocks\b",
    r"\b\d+ sequential attention layers\b",
    r"\bSHA-\d+\b",
    r"\bAES\b",
    r"\bChaCha20-Poly1305\b",
    r"\bArgon2\b",
    r"\bPBKDF2\b",
    r"\bECDSA\b",
    r"\bEd25519\b",
    r"\bBLAKE2\b",
    r"\bJWTs?\b",
    r"\bPKCE\b",
    r"\bHTTP/3\b",
    r"\bHTTP/2\b",
    r"\bQUIC\b",
    r"\bTCP\b",
    r"\bUDP\b",
    r"\bTLS 1\.\d\b",
    r"\bPostgreSQL\b",
    r"\bMySQL(?: InnoDB)?\b",
    r"\bMongoDB\b",
    r"\bRedis\b",
    r"\bMemcached\b",
    r"\bCassandra\b",
    r"\bRocksDB\b",
    r"\bRaft\b",
    r"\bMulti-Paxos\b",
    r"\bAdamW\b",
    r"\bSGD\b",
    r"\bB\+ Tree\b",
    r"\bLSM trees?\b",
    r"\bWAL\b",
    r"\bBGP\b",
    r"\bOSPF\b",
    r"\bSWIM\b",
    r"\bConsul\b",
    r"\betcd\b",
    r"\bCockroachDB\b",
    r"\bDynamo\b",
    r"\bRiak KV\b",
]


def is_prompt_injection(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def abstain_response():
    return {
        "answer": None,
        "sources": [],
        "scores": [],
        "confidence": 0.0,
    }


def extract_answer(question: str, passage: str):
    question_lower = question.lower()

    for pattern in FACTUAL_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            passage_match = re.search(pattern, passage, re.IGNORECASE)
            if passage_match:
                return passage_match.group(0)

    sentences = re.split(r"(?<=[.!?])\s+", passage)

    keywords = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+\-.]*", question)
        if word.lower() not in STOPWORDS
    ]

    best_sentence = None
    best_score = -1

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0

        for keyword in keywords:
            if keyword in sentence_lower:
                score += 2

        if any(
            word in sentence_lower
            for word in ["is", "are", "uses", "requires", "provides", "implements", "relies"]
        ):
            score += 1

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence.strip() if best_sentence else None


@app.post("/query")
def query(request: QueryRequest):
    question = (request.question or "").strip()

    if not question or is_prompt_injection(question):
        return abstain_response()

    top_k = max(1, min(request.top_k or 3, 10))

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    scores = passage_embeddings @ question_embedding
    ranked = np.argsort(scores)[::-1][:top_k]

    sources = [passage_ids[i] for i in ranked]
    passage_scores = [round(float(scores[i]), 4) for i in ranked]

    best_score = passage_scores[0]
    second_score = passage_scores[1] if len(passage_scores) > 1 else 0.0

    # Be permissive: answer unless retrieval is clearly bad.
    if best_score < 0.30:
        return abstain_response()

    answer = extract_answer(question, passage_texts[ranked[0]])

    if not answer:
        return abstain_response()

    confidence = min(0.98, max(0.0, (best_score - 0.20) / 0.60))

    return {
        "answer": answer,
        "sources": sources,
        "scores": passage_scores,
        "confidence": round(confidence, 4),
    }


@app.get("/")
def health():
    return {"status": "ok"}
