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
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# Retrieval
TOP_K = 20
FINAL_K_FACT = 4
FINAL_K_SUMMARY = 10

SIMILARITY_THRESHOLD = 0.32
CONFIDENCE_THRESHOLD = 0.46

# Context
MAX_CONTEXT_CHARS = 4000
MAX_CONTEXT_BLOCK_CHARS = 800

# Generation
MAX_TOKENS_FACT = 800
MAX_TOKENS_SUMMARY = 1000

MODEL_NAME = "gemma3:1b"

TEMPERATURE = 0.1
STREAM = True

# Runtime
OLLAMA_CTX = 4096
CPU_THREADS_USED = 8

# Routing
RAG_MIN_RESULTS = 1
RAG_MIN_CONTEXT_WORDS = 20


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

    return Path(path).read_text(
        encoding="utf-8"
    )


def clean_text(text: str) -> str:
    """
    Remove noisy formatting.
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

    return re.split(
        r'(?<=[.!?])\s+',
        text
    )


def chunk_text(text: str):

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

    texts = []
    sections = []

    for section_title, body in sectioned_text:

        for chunk in chunk_text(body):

            cleaned = chunk.strip()

            if not cleaned:
                continue

            texts.append(cleaned)
            sections.append(section_title)

    return texts, sections


# =========================================================
# FAISS
# =========================================================

def build_faiss_index(texts):

    global faiss_index

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
# EMBEDDING CACHE
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
# QUERY CLASSIFICATION
# =========================================================

def classify_query(query: str):

    q = query.lower()

    # CHAPTER MODE (highest priority for structure)
    if any(x in q for x in [
        "chapter",
        "what is written in chapter",
        "summarise chapter",
        "summary of chapter"
    ]):
        return "chapter"

    # ENTITY FACT QUESTIONS
    if any(x in q for x in [
        "who was",
        "who is",
        "who were",
        "what is torah",
        "what is hillel",
        "what is akiva"
    ]):
        return "entity"

    # CONCEPTUAL / EXPLANATION
    if any(x in q for x in [
        "what is",
        "define",
        "meaning",
        "why",
        "how",
        "explain"
    ]):
        return "concept"

    # SUMMARY REQUESTS
    if any(x in q for x in [
        "summarise",
        "summary",
        "overview"
    ]):
        return "summary"

    return "general"

# =========================================================
# QUERY EXPANSION
# =========================================================

def expand_query(query: str):

    q = query.lower()

    expansions = {

        "great assembly":
            "great synagogue men of the great synagogue",

        "great synagogue":
            "great assembly",

        "torah":
            "law teachings pentateuch",

        "akiva":
            "rabbi akiva tannaim",

        "hillel":
            "rabbi hillel",

        "shammai":
            "rabbi shammai",

        "chapter 1":
            "chapter i",

        "chapter 2":
            "chapter ii",

        "chapter 3":
            "chapter iii",

        "chapter 4":
            "chapter iv",

        "chapter 5":
            "chapter v",

        "chapter 6":
            "chapter vi"
    }

    expanded = query

    for key, value in expansions.items():

        if key in q:
            expanded += " " + value

    return expanded


# =========================================================
# RETRIEVAL HELPERS
# =========================================================

STOPWORDS = {
    "the", "a", "an", "of", "to", "and",
    "what", "who", "was", "were", "is",
    "are", "did", "does", "in"
}


def lexical_overlap_score(query_words, text_lower):

    matches = sum(
        1
        for word in query_words
        if word in text_lower
    )

    score = matches * 0.06

    if matches >= 2:
        score += 0.12

    if matches >= 3:
        score += 0.18

    return score


def explanation_boost(text_lower):

    patterns = [

        "their work was",
        "their role was",
        "they interpreted",
        "they taught",
        "they developed",
        "they instituted",
        "served as",
        "constituted",
        "known as",
        "refers to"
    ]

    if any(p in text_lower for p in patterns):
        return 0.18

    return 0.0


def factual_density_boost(text):

    if 180 <= len(text) <= 650:
        return 0.10

    return 0.0


def commentary_penalty(text_lower):

    commentary_markers = [

        "translator",
        "editor",
        "footnote",
        "appendix",
        "maimonides says",
        "others have written"
    ]

    if any(x in text_lower for x in commentary_markers):
        return -0.22

    return 0.0


# =========================================================
# CHAPTER RETRIEVAL
# =========================================================

def retrieve_chapter(chapter_query: str):
    """
    Direct section retrieval for chapter summaries.
    """

    q = chapter_query.lower()

    roman_map = {
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
        "5": "V",
        "6": "VI"
    }

    for num, roman in roman_map.items():

        if f"chapter {num}" in q:

            target = f"CHAPTER {roman}"

            results = []

            for text, section in zip(
                chunk_texts,
                chunk_sections
            ):

                if target in section:

                    results.append({
                        "text": text,
                        "section": section,
                        "score": 1.0
                    })

            return results[:FINAL_K_SUMMARY]

    return None

def entity_safety_filter(query, text_lower):

    q = query.lower()

    # prevents known hallucinated conflations
    if "who are the jews" in q:

        if "great synagogue" in text_lower:
            return -0.4

        if "men of the great assembly" in text_lower:
            return -0.4

    return 0.0

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(query: str, query_type: str):

    # =========================
    # CHAPTER MODE (HARD ROUTE)
    # =========================
    if query_type == "chapter":

        chapter_hits = retrieve_chapter(query)

        if chapter_hits:
            return chapter_hits

    expanded = expand_query(query)
    q_emb = embed_query(expanded)

    top_k = TOP_K

    if query_type == "summary":
        top_k = 30  # IMPORTANT: summaries need breadth

    scores, ids = faiss_index.search(
        np.array([q_emb]),
        top_k
    )

    results = []
    seen = set()

    query_words = {
        w for w in re.findall(r"\w+", query.lower())
        if w not in STOPWORDS
    }

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        text = chunk_texts[idx].strip()
        text_lower = text.lower()

        if text in seen:
            continue
        seen.add(text)

        final_score = float(score)

        final_score += lexical_overlap_score(query_words, text_lower)
        final_score += explanation_boost(text_lower)
        final_score += factual_density_boost(text)
        final_score += commentary_penalty(text_lower)
        final_score += entity_safety_filter(query, text_lower)

        if final_score < SIMILARITY_THRESHOLD:
            continue

        results.append({
            "text": text,
            "section": chunk_sections[idx],
            "score": round(final_score, 3)
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    if query_type == "summary":
        return results[:FINAL_K_SUMMARY]

    return results[:FINAL_K_FACT]

# =========================================================
# CONTEXT
# =========================================================

def compress_context(results):

    blocks = []
    seen = set()

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

def build_rag_prompt(context, query, query_type):

    base_rules = """
You are answering using a structured historical/religious text.

CRITICAL RULES:
- Use ALL relevant context provided
- Do NOT quote randomly
- Do NOT focus on a single passage
- Do NOT confuse entities (e.g. groups vs individuals)
- Synthesize information across passages
"""

    if query_type == "chapter":

        instruction = """
You are writing a CHAPTER SYNTHESIS.

Requirements:
- Organize into themes
- Combine all teachings
- Mention key figures only when relevant
- Do NOT quote isolated lines
- Ensure full coverage of the chapter content
- Write 3–6 paragraphs minimum
"""

    elif query_type == "summary":

        instruction = """
You are writing a DETAILED SUMMARY.

Requirements:
- Cover ALL major ideas in the context
- Include multiple rabbis and teachings
- Explain ethical themes clearly
- Avoid quoting single sentences
- Prefer synthesis over repetition
- Use multiple paragraphs if needed
"""

    elif query_type == "entity":

        instruction = """
You are answering a factual identity question.

Requirements:
- Be precise
- Use context if relevant
- Do NOT conflate groups or roles
- Avoid assumptions beyond text
"""

    else:

        instruction = """
Explain clearly and naturally.

Requirements:
- Use context if helpful
- Keep explanation accurate
- Avoid hallucinations
"""

    return f"""
{base_rules}

{instruction}

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
""".strip()

# =========================================================
# GENERATION
# =========================================================

def generate(prompt: str, max_tokens: int):

    return ollama.chat(

        model=MODEL_NAME,

        stream=STREAM,

        options={

            "temperature": TEMPERATURE,

            "num_predict": max_tokens,

            "num_ctx": OLLAMA_CTX,

            "num_thread": CPU_THREADS_USED,

            "repeat_penalty": 1.08,

            "top_k": 30,

            "top_p": 0.9
        },

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


# =========================================================
# DIRECT LLM
# =========================================================

def direct_llm_answer(query: str, retrieved_context: str | None = None):

    system = """
You are a neutral factual assistant.

STYLE RULES:
- Do not moralise or lecture
- Do not correct the user emotionally
- Do not assume intent
- If question is offensive, respond factually and neutrally
- If context is missing, answer generally and carefully
"""

    context_block = ""

    if retrieved_context:
        context_block = f"""
Context (use if relevant):
{retrieved_context}
"""

    prompt = f"""
{system}

{context_block}

Question:
{query}

Answer:
""".strip()

    response = ollama.chat(
        model=MODEL_NAME,
        stream=False,
        options={
            "temperature": 0.3,
            "num_predict": 220,
            "num_ctx": OLLAMA_CTX,
            "num_thread": CPU_THREADS_USED
        },
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# =========================================================
# ROUTING
# =========================================================

def should_use_rag(query: str):

    q = query.lower().strip()

    non_rag = [

        "hello",
        "hi",
        "how are you",
        "tell me a joke"
    ]

    if q in non_rag:
        return False

    return True


def is_hostile(query: str):

    q = query.lower()

    patterns = [
        "are jews evil",
        "are jews greedy",
        "jews are evil",
        "jews are greedy",
        "zionists are evil"
    ]

    return any(p in q for p in patterns)


# =========================================================
# ASK
# =========================================================

def ask(query: str):

    query = query.strip()

    if not query:
        return

    print("\n--- ANSWER ---\n")

    if is_hostile(query):
        # still allows factual answer, just no moral framing
        answer = direct_llm_answer(query)
        print(answer)
        return

    if not should_use_rag(query):
        print(direct_llm_answer(query))
        return

    query_type = classify_query(query)

    results = retrieve(query, query_type)

    if not results:
        print(direct_llm_answer(query))
        return

    context = compress_context(results)

    if len(context.split()) < RAG_MIN_CONTEXT_WORDS:
        print(direct_llm_answer(query))
        return

    prompt = build_rag_prompt(context, query, query_type)

    max_tokens = (
        MAX_TOKENS_SUMMARY
        if query_type in ["summary", "chapter"]
        else MAX_TOKENS_FACT
    )

    response = generate(prompt, max_tokens)

    full = ""

    for chunk in response:
        if "message" not in chunk:
            continue

        content = chunk["message"].get("content", "")
        print(content, end="", flush=True)
        full += content

    print()
    return full

# =========================================================
# WARMUP
# =========================================================

def warmup():

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

        print(f"Warmup failed: {error}")


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