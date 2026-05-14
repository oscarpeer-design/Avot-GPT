from pathlib import Path
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
from functools import lru_cache

# =========================================================
# CONFIG
# =========================================================

DOCUMENT_PATH = r"C:\Users\Oscar\source\repos\Avot-GPT\PerkeiAvot.txt"

CHUNK_SIZE = 320
CHUNK_OVERLAP = 80

TOP_K = 18
FINAL_K = 4

SIMILARITY_THRESHOLD = 0.33

MAX_CONTEXT_CHARS = 1600
MAX_TOKENS = 110

MODEL_NAME = "phi3"
TEMPERATURE = 0.1
STREAM = True

RAG_MIN_RESULTS = 2
RAG_MIN_CONTEXT_WORDS = 35


# =========================================================
# MODEL
# =========================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# GLOBAL STATE
# =========================================================

chunk_texts = []
chunk_sections = []
faiss_index = None


# =========================================================
# LOAD / CLEAN
# =========================================================

def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def clean_text(text):
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================================================
# SECTION SPLIT
# =========================================================

SECTION_PATTERN = r'(CHAPTER\s+[IVXLC\d]+\.?)'


def split_sections(text):
    parts = re.split(SECTION_PATTERN, text)

    sections = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        sections.append((parts[i].strip(), parts[i + 1].strip()))

    return sections


# =========================================================
# CHUNKING (WITH OVERLAP)
# =========================================================

def chunk_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    buffer = []
    size = 0

    for s in sentences:
        if size + len(s) <= CHUNK_SIZE:
            buffer.append(s)
            size += len(s)
        else:
            yield " ".join(buffer)

            buffer = buffer[-2:]  # overlap
            buffer.append(s)
            size = sum(len(x) for x in buffer)

    if buffer:
        yield " ".join(buffer)


def build_chunks(sectioned):
    texts, sections = [], []

    for section_title, text in sectioned:
        for chunk in chunk_text(text):
            texts.append(chunk)
            sections.append(section_title)

    return texts, sections


# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(texts):
    global faiss_index

    if not texts:
        raise ValueError("No chunks generated")

    print("Embedding chunks...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)


# =========================================================
# QUERY EMBEDDING CACHE
# =========================================================

@lru_cache(maxsize=256)
def embed_query(query):
    return embedding_model.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(query):
    q = embed_query(query)

    scores, ids = faiss_index.search(
        np.array([q]),
        TOP_K
    )

    results = []
    query_l = query.lower()

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        text = chunk_texts[idx]
        text_l = text.lower()

        # base semantic score
        final_score = float(score)

        # =====================================================
        # BOOST 1: exact entity match
        # =====================================================
        if any(word in text_l for word in query_l.split()):
            final_score += 0.08

        # =====================================================
        # BOOST 2: definition detection (CRITICAL FIX)
        # =====================================================
        definition_triggers = [
            "is", "are", "refers to", "was", "were", "known as"
        ]

        if any(t in text_l for t in definition_triggers):
            final_score += 0.12

        # =====================================================
        # FILTER
        # =====================================================
        if final_score < SIMILARITY_THRESHOLD:
            continue

        results.append({
            "text": text,
            "section": chunk_sections[idx],
            "score": final_score
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:FINAL_K]


# =========================================================
# CONTEXT
# =========================================================

def compress_context(results):
    blocks = []

    for r in results:
        block = f"[{r['section']}]\n{r['text']}"
        blocks.append(block[:700])

    return "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]


# =========================================================
# PROMPT
# =========================================================

def build_prompt(context, query):
    return f"""
You are a strict QA system.

RULES:
- Use ONLY context
- If answer is missing, say "I don't know"
- Do not infer

Context:
{context}

Question:
{query}

Answer:
""".strip()


# =========================================================
# LLM
# =========================================================

def generate(prompt, stream=True):
    return ollama.chat(
        model=MODEL_NAME,
        stream=stream,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        },
        messages=[{"role": "user", "content": prompt}]
    )


# =========================================================
# DIRECT MODE (FAST PATH)
# =========================================================

def direct_llm_answer(query):
    prompt = f"""
Answer clearly and directly.

Question:
{query}

Answer:
""".strip()

    res = ollama.chat(
        model=MODEL_NAME,
        stream=False,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        },
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]


# =========================================================
# ROUTER (CORE FIX)
# =========================================================

def should_use_rag(query):
    q = query.lower()

    strong_rag_signals = [
        "according to", "in the text", "chapter", "mentioned", "who were"
    ]

    weak_queries = [
        "what is", "who are", "explain", "define"
    ]

    if any(x in q for x in weak_queries):
        return False

    if any(x in q for x in strong_rag_signals):
        return True

    return True  # default safe


def route(query):
    results = retrieve(query)

    # FAST PATH (no LLM at all)
    if len(results) == 0:
        return direct_llm_answer(query)

    # LOW CONFIDENCE → skip RAG entirely
    if len(results) < 2 or results[0]["score"] < 0.40:
        return direct_llm_answer(query)

    context = compress_context(results)

    print("=== CONTEXT ===")
    print(context)

    if len(context.split()) < 30:
        return direct_llm_answer(query)

    prompt = build_prompt(context, query)

    return generate(prompt, stream=STREAM)


# =========================================================
# ASK
# =========================================================

def ask(query):
    response = route(query)

    print("\n--- ANSWER ---\n")

    if isinstance(response, dict) and "message" in response:
        print(response["message"]["content"])
        return

    for chunk in response:
        print(chunk["message"]["content"], end="", flush=True)


# =========================================================
# WARMUP
# =========================================================

def warmup():
    print("Warming up model...")

    ollama.chat(
        model=MODEL_NAME,
        stream=False,
        messages=[{
            "role": "user",
            "content": "Say ready"
        }]
    )

    print("Model warm.")


# =========================================================
# MAIN
# =========================================================

def main():
    raw = load_text(DOCUMENT_PATH)
    cleaned = clean_text(raw)

    sections = split_sections(cleaned)
    texts, secs = build_chunks(sections)

    global chunk_texts, chunk_sections
    chunk_texts = texts
    chunk_sections = secs

    build_faiss_index(texts)
    warmup()

    print("\nSystem ready\n")

    while True:
        q = input("\nAsk: ").strip()
        if q.lower() == "exit":
            break
        ask(q)


if __name__ == "__main__":
    main()