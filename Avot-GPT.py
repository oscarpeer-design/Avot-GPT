from pathlib import Path
from functools import lru_cache

import re
import numpy as np
import faiss
import ollama

from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

DOCUMENT_PATH = r"C:\Users\Oscar\source\repos\Avot-GPT\PerkeiAvot.txt"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40

# Retrieval
TOP_K = 15
FINAL_K = 3

SIMILARITY_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = 0.48

# Context
MAX_CONTEXT_CHARS = 2000
MAX_CONTEXT_BLOCK_CHARS = 300

# Generation
MAX_TOKENS = 350

MODEL_NAME = "gemma3:1b"

TEMPERATURE = 0.05
STREAM = True

# Runtime
OLLAMA_CTX = 2048
CPU_THREADS_USED = 8

# Routing
RAG_MIN_RESULTS = 1
RAG_MIN_CONTEXT_WORDS = 18
RAG_MIN_CONTEXT_WORDS = 18

# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# GLOBAL STATE
# =========================================================

chunk_texts = []
chunk_sections = []

faiss_index = None

# =========================================================
# LOAD + CLEAN
# =========================================================

def load_text(path: str) -> str:
    """
    Load UTF-8 document text.
    """

    return Path(path).read_text(
        encoding="utf-8"
    )


def clean_text(text: str) -> str:
    """
    Remove noisy formatting that harms embeddings.
    """

    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================
# SECTION SPLITTING
# =========================================================

SECTION_PATTERN = r"(CHAPTER\s+[IVXLC\d]+\.?)"


def split_sections(text: str):
    """
    Split corpus into:
    (section_title, section_body)
    """

    parts = re.split(
        SECTION_PATTERN,
        text
    )

    sections = []

    for i in range(1, len(parts), 2):

        if i + 1 >= len(parts):
            break

        title = parts[i].strip()
        body = parts[i + 1].strip()

        sections.append(
            (title, body)
        )

    return sections

# =========================================================
# CHUNKING
# =========================================================

def split_sentences(text: str):
    """
    Sentence-aware splitting.
    """

    return re.split(
        r'(?<=[.!?])\s+',
        text
    )


def chunk_text(text: str):
    """
    Sliding-window chunking with overlap.
    """

    sentences = split_sentences(text)

    buffer = []
    current_size = 0

    for sentence in sentences:

        sentence_len = len(sentence)

        if current_size + sentence_len <= CHUNK_SIZE:

            buffer.append(sentence)
            current_size += sentence_len

            continue

        if buffer:
            yield " ".join(buffer)

        overlap_buffer = []

        overlap_size = 0

        for old_sentence in reversed(buffer):

            overlap_buffer.insert(0, old_sentence)

            overlap_size += len(old_sentence)

            if overlap_size >= CHUNK_OVERLAP:
                break

        buffer = overlap_buffer + [sentence]

        current_size = sum(
            len(x) for x in buffer
        )

    if buffer:
        yield " ".join(buffer)


def build_chunks(sectioned_text):
    """
    Build aligned:
    - chunk_texts
    - chunk_sections
    """

    texts = []
    sections = []

    for section_title, body in sectioned_text:

        for chunk in chunk_text(body):

            cleaned_chunk = chunk.strip()

            if not cleaned_chunk:
                continue

            texts.append(cleaned_chunk)
            sections.append(section_title)

    return texts, sections

# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(texts):
    """
    Build cosine-similarity FAISS index.
    """

    global faiss_index

    if not texts:
        raise ValueError(
            "No chunks generated."
        )

    print("Embedding chunks...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(
        dimension
    )

    faiss_index.add(embeddings)

# =========================================================
# QUERY EMBEDDING CACHE
# =========================================================

@lru_cache(maxsize=256)
def embed_query(query: str):

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    return np.asarray(
        embedding,
        dtype=np.float32
    )

# =========================================================
# RETRIEVAL
# =========================================================

STOPWORDS = {
    "the", "a", "an", "of", "to", "and",
    "what", "who", "was", "were", "is",
    "are", "did", "does", "in"
}


def lexical_overlap_score(query_words, text_lower):
    """
    Exact keyword overlap boost.
    """

    matches = sum(
        1
        for word in query_words
        if word in text_lower
    )

    score = matches * 0.07

    # Strong boost for multi-word matches
    if matches >= 2:
        score += 0.14

    if matches >= 3:
        score += 0.20

    return score


def definition_boost(text_lower):
    """
    Boost definition-style passages.
    """

    patterns = [
        "was",
        "were",
        "is",
        "are",
        "consisted of",
        "known as",
        "refers to"
    ]

    if any(p in text_lower for p in patterns):
        return 0.12

    return 0.0


def explanation_boost(text_lower):
    """
    HUGE quality improvement:
    boosts role/explanation passages.
    """

    patterns = [
        "their work was",
        "their role was",
        "they interpreted",
        "they taught",
        "they developed",
        "they instituted",
        "they established",
        "they were ascribed",
        "served as",
        "depositaries of",
        "constituted"
    ]

    if any(p in text_lower for p in patterns):
        return 0.22

    return 0.0


def factual_density_boost(text):
    """
    Prefer medium factual passages.
    """

    text_len = len(text)

    # Best range for factual synthesis
    if 180 <= text_len <= 450:
        return 0.12

    return 0.0


def commentary_penalty(text_lower):
    """
    Penalize commentary-heavy sections.
    """

    commentary_markers = [
        "maimonides",
        "blessed memory",
        "others have written",
        "supports his view",
        "this verse",
        "divine inspiration"
    ]

    if any(c in text_lower for c in commentary_markers):
        return -0.28

    return 0.0


def retrieve(query: str):

    expanded_query = expand_query(query)

    query_embedding = embed_query(expanded_query)

    scores, ids = faiss_index.search(
        np.array([query_embedding]),
        TOP_K
    )

    query_words = {
        w for w in re.findall(r"\w+", query.lower())
        if w not in STOPWORDS
    }

    results = []

    seen_texts = set()

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        text = chunk_texts[idx].strip()

        # deduplicate repeated chunks
        if text in seen_texts:
            continue

        seen_texts.add(text)

        text_lower = text.lower()

        final_score = float(score)

        # =====================================
        # RERANKING
        # =====================================

        final_score += lexical_overlap_score(
            query_words,
            text_lower
        )

        final_score += definition_boost(
            text_lower
        )

        final_score += explanation_boost(
            text_lower
        )

        final_score += factual_density_boost(
            text
        )

        final_score += commentary_penalty(
            text_lower
        )

        # =====================================
        # FILTER
        # =====================================

        if final_score < SIMILARITY_THRESHOLD:
            continue

        results.append({
            "text": text,
            "section": chunk_sections[idx],
            "score": round(final_score, 3)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:FINAL_K]

# =========================================================
# CONTEXT COMPRESSION
# =========================================================

def compress_context(results):
    """
    Build concise high-signal context.
    """

    seen = set()

    blocks = []

    for result in results:

        text = result["text"].strip()

        if text in seen:
            continue

        seen.add(text)

        if len(text) > MAX_CONTEXT_BLOCK_CHARS:
            text = text[:MAX_CONTEXT_BLOCK_CHARS]

        block = (
            f"[{result['section']}]\n"
            f"{text}"
        )

        blocks.append(block)

    context = "\n\n".join(blocks)

    return context[:MAX_CONTEXT_CHARS]

# =========================================================
# PROMPTS
# =========================================================

def build_rag_prompt(context, query):

    return f"""
You are a precise but thorough QA system.

CRITICAL INSTRUCTIONS:
- Use ONLY the provided context
- Do NOT invent information
- If the context contains multiple facts, include ALL of them
- Combine related ideas into a complete explanation
- Write full, concise sentences only

FORMAT:
- Write 2–4 sentences depending on available information
- If listing people, roles, or groups: include ALL mentioned roles
- If describing a concept: explain definition + role + context

Context:
{context}

Question:
{query}

Answer:
""".strip()


def build_direct_prompt(query: str):
    """
    General assistant prompt.
    """

    return f"""
Answer clearly and directly.

Question:
{query}

Answer:
""".strip()

# =========================================================
# OLLAMA
# =========================================================

def ollama_generate(prompt: str):
    """
    Fast low-latency generation.
    """

    return ollama.chat(
        model=MODEL_NAME,

        stream=STREAM,

        options={

            "temperature": TEMPERATURE,

            "num_predict": MAX_TOKENS,

            "num_ctx": OLLAMA_CTX,

            "num_thread": CPU_THREADS_USED,

            "repeat_penalty": 1.05,

            "top_k": 20,

            "top_p": 0.8
        },

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

# =========================================================
# ROUTING
# =========================================================

def should_use_rag(query: str) -> bool:

    q = query.lower().strip()

    non_rag = [
        "hello",
        "hi",
        "how are you",
        "tell me a joke"
    ]

    if any(x == q for x in non_rag):
        return False

    return True

# =========================================================
# ANSWERING
# =========================================================

def direct_llm_answer(query: str):

    prompt = f"""
Answer clearly and accurately.

Question:
{query}

Answer:
""".strip()

    response = ollama.chat(
        model=MODEL_NAME,
        stream=False,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,

            "num_ctx": OLLAMA_CTX,
            "num_thread": CPU_THREADS_USED
        },
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def rag_answer(query: str):
    """
    Full RAG pipeline.
    """

    results = retrieve(query)

    if not results:
        return (
            "No relevant context found."
        )

    if results[0]["score"] < CONFIDENCE_THRESHOLD:
        return direct_llm_answer(query)

    context = compress_context(results)

    if len(context.split()) < 25:
        return direct_llm_answer(query)

    prompt = build_rag_prompt(
        context,
        query
    )

    response = ollama_generate(prompt)

    return response["message"]["content"]

# =========================================================
# GENERATE
# =========================================================

def generate(prompt: str):
    """
    Unified Ollama generation.

    Always returns:
    - string if STREAM=False
    - generator if STREAM=True
    """

    response = ollama.chat(
        model=MODEL_NAME,
        stream=STREAM,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,

            # Performance
            "num_ctx": OLLAMA_CTX,
            "num_thread": CPU_THREADS_USED,

            # Stability
            "repeat_penalty": 1.05,
            "top_k": 20,
            "top_p": 0.8
        },
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response


# =========================================================
# ASK
# =========================================================

def expand_query(query: str) -> str:
    """
    Expand queries into corpus language.
    Greatly improves retrieval quality.
    """

    q = query.lower()

    expansions = {
        "jews": "israel torah jewish people rabbis",
        "jewish": "israel torah rabbinical",
        "god": "lord heaven divine",
        "afterlife": "world to come gehinnom",
        "women": "wife woman",
        "great assembly": "great synagogue men of the great synagogue",
        "hillel": "rabbi hillel",
        "shammai": "rabbi shammai",
        "torah": "law teachings",
    }

    expanded = query

    for key, value in expansions.items():
        if key in q:
            expanded += " " + value

    return expanded

def ask(query: str):

    query = query.strip()
    query = expand_query(query)

    if not query:
        print("Please enter a question.")
        return

    print("\n--- ANSWER ---\n")

    # =====================================================
    # DIRECT MODE
    # =====================================================

    if not should_use_rag(query):

        answer = direct_llm_answer(query)

        print(answer)
        return

    # =====================================================
    # RETRIEVAL
    # =====================================================

    results = retrieve(query)

    if (
        len(results) < RAG_MIN_RESULTS
        or results[0]["score"] < CONFIDENCE_THRESHOLD
    ):
        answer = direct_llm_answer(query)

        print(answer)
        return

    context = compress_context(results)

    if len(context.split()) < RAG_MIN_CONTEXT_WORDS:

        answer = direct_llm_answer(query)

        print(answer)
        return

    prompt = build_rag_prompt(context, query)

    response = generate(prompt)

    # =====================================================
    # STREAMING MODE
    # =====================================================

    if STREAM:

        full_response = ""

        for chunk in response:

            if "message" not in chunk:
                continue

            content = chunk["message"].get("content", "")

            print(content, end="", flush=True)

            full_response += content

        print()

        return full_response

    # =====================================================
    # NON-STREAM MODE
    # =====================================================

    else:

        print(response["message"]["content"])

        return response["message"]["content"]

# =========================================================
# WARMUP
# =========================================================

def warmup():
    """
    Warm model into RAM.
    """

    print("Warming up model...")

    try:

        ollama.chat(
            model=MODEL_NAME,

            stream=False,

            messages=[
                {
                    "role": "user",
                    "content": "ready"
                }
            ],

            options={
                "num_predict": 5
            }
        )

        print("Model warm.")

    except Exception as error:

        print(
            f"Warmup failed: {error}"
        )

        print(
            "Check Ollama installation."
        )

# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading text...")

    raw_text = load_text(
        DOCUMENT_PATH
    )

    cleaned_text = clean_text(
        raw_text
    )

    sectioned_text = split_sections(
        cleaned_text
    )

    texts, sections = build_chunks(
        sectioned_text
    )

    global chunk_texts
    global chunk_sections

    chunk_texts = texts
    chunk_sections = sections

    build_faiss_index(texts)

    warmup()

    print("\nSystem ready.\n")

    while True:

        query = input(
            "\nAsk: "
        ).strip()

        if query.lower() == "exit":
            break

        ask(query)

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()