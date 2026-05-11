from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import re
import numpy as np

#Text cleaning
def clean_text(text):
    # remove bracket citations like [47], [s], [12a]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # remove parenthetical numeric citations like (47), (50)
    text = re.sub(r'\(\d+\)', '', text)

    # remove “To understand more...” style phrases
    text = re.sub(r'To see more[^.]*\.', '', text, flags=re.IGNORECASE)

    # remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -----------------------------
# CONFIG
# -----------------------------
CHUNK_SIZE = 500
TOP_K = 8
MIN_CONTEXT_WORDS = 40

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# CHROMA DB
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")

# Reset collection to avoid duplicates (IMPORTANT)
try:
    client.delete_collection("documents")
except:
    pass

collection = client.get_or_create_collection("documents")

# -----------------------------
# LOAD DOCUMENT
# -----------------------------
file_path = Path(r"C:\Users\Oscar\source\repos\Avot-GPT\PerkeiAvot.txt")
text = file_path.read_text(encoding="utf-8")

# -----------------------------
# CLEAN TEXT (IMPORTANT)
# -----------------------------
text = clean_text(text)

# -----------------------------
# SENTENCE CHUNKING
# -----------------------------
sentences = re.split(r'(?<=[.!?])\s+', text)

chunks = []
current = []
length = 0

for s in sentences:
    if length + len(s) < CHUNK_SIZE:
        current.append(s)
        length += len(s)
    else:
        chunks.append(" ".join(current))
        current = [s]
        length = len(s)

if current:
    chunks.append(" ".join(current))

# -----------------------------
# INDEXING
# -----------------------------
for i, chunk in enumerate(chunks):
    emb = embed_model.encode(chunk)
    emb = emb / np.linalg.norm(emb)  # normalize embeddings

    collection.add(
        documents=[chunk],
        embeddings=[emb.tolist()],
        ids=[str(i)]
    )

print("Document indexed successfully.")

# -----------------------------
# RETRIEVAL FUNCTION (IMPROVED)
# -----------------------------
def retrieve(query):
    q_emb = embed_model.encode(query)
    q_emb = q_emb / np.linalg.norm(q_emb)

    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=TOP_K
    )

    docs = results["documents"][0]

    # FILTER: remove weak / tiny chunks
    filtered = [
        d for d in docs
        if len(d.split()) > 20
    ]

    # SIMPLE RERANKING (similarity proxy)
    scored = []
    for d in filtered:
        d_emb = embed_model.encode(d)
        d_emb = d_emb / np.linalg.norm(d_emb)
        score = np.dot(q_emb, d_emb)
        scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [d for _, d in scored[:3]]

# -----------------------------
# PROMPT
# -----------------------------
def build_prompt(context, query):
    return f"""You are a strict retrieval-based assistant.

RULES:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- If the answer is not in the context, say:
  "I don't know based on the provided context."
- Do not guess or infer.
- Be concise and factual.

Context:
{context}

Question:
{query}

Answer:
"""

# -----------------------------
# CHAT FUNCTION
# -----------------------------
def chat(query):
    chunks = retrieve(query)

    context = "\n".join(chunks)

    print("\n--- CONTEXT USED ---\n")
    print(context)

    # SAFETY CHECK (CRITICAL)
    if len(context.split()) < MIN_CONTEXT_WORDS:
        print("\nNot enough relevant context to answer safely.")
        return

    prompt = build_prompt(context, query)

    response = ollama.chat(
        model="phi3",
        stream=True,
        options={
            "num_ctx": 2048,
            "num_predict": 150,
            "temperature": 0.2
        },
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n")

    for chunk in response:
        print(chunk["message"]["content"], end="", flush=True)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break
        chat(query)