from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import re
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================

DOCUMENT_PATH = (
    r"C:\Users\Oscar\source\repos\Avot-GPT\PerkeiAvot.txt"
)

CHROMA_DB_PATH = "./chroma_db"

COLLECTION_NAME = "documents"

SECTION_PATTERN = r'(CHAPTER\s+[IVXLC\d]+\.?)'

CHUNK_SIZE = 300

TOP_K_RETRIEVAL_RESULTS = 25 # Keep high and then reduce later

FINAL_RERANKED_RESULTS = 5

MINIMUM_CHUNK_WORD_COUNT = 25

MINIMUM_CONTEXT_WORD_COUNT = 10

OLLAMA_MODEL_NAME = "phi3"

OLLAMA_TEMPERATURE = 0.1 # Keep 0.05 to 0.2 to prevent hallucinations

OLLAMA_CONTEXT_WINDOW = 2048

OLLAMA_MAX_TOKENS = 200

SIMILARITY_THRESHOLD = 0.2 # verify relevance

# =========================================================
# MODEL INITIALIZATION
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_vector_database():

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    try:
        chroma_client.delete_collection(
            COLLECTION_NAME
        )

    except:
        pass

    document_collection = (
        chroma_client.get_or_create_collection(
            COLLECTION_NAME
        )
    )

    return document_collection


# =========================================================
# DOCUMENT LOADING
# =========================================================

def load_document_text(document_path):

    return Path(document_path).read_text(
        encoding="utf-8"
    )

# =========================================================
# TEXT SANITIZATION
# =========================================================

def sanitize_document_text(raw_document_text):

    # Remove bracket citations
    raw_document_text = re.sub(
        r'\[[^\]]*\]',
        '',
        raw_document_text
    )

    # Remove parenthetical citations
    raw_document_text = re.sub(
        r'\(\d+\)',
        '',
        raw_document_text
    )

    # Remove editorial phrases
    raw_document_text = re.sub(
        r'To see more[^.]*\.',
        '',
        raw_document_text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace
    raw_document_text = re.sub(
        r'\s+',
        ' ',
        raw_document_text
    )

    return raw_document_text.strip()


# =========================================================
# SECTION DETECTION
# =========================================================

def extract_document_sections(
    sanitized_document_text,
    section_pattern
):

    split_sections = re.split(
        section_pattern,
        sanitized_document_text
    )

    structured_sections = []

    for index in range(
        1,
        len(split_sections),
        2
    ):

        section_title = (
            split_sections[index].strip()
        )

        section_text = (
            split_sections[index + 1].strip()
        )

        structured_sections.append({
            "title": section_title,
            "text": section_text
        })

    return structured_sections


# =========================================================
# SECTION-AWARE CHUNKING
# =========================================================

def build_section_aware_chunks(
    structured_sections,
    chunk_size
):

    section_aware_chunks = []

    chunk_id = 0

    for section in structured_sections:

        section_title = section["title"]

        section_text = section["text"]

        section_sentences = re.split(
            r'(?<=[.!?])\s+',
            section_text
        )

        current_chunk_sentences = []

        current_chunk_length = 0

        for sentence in section_sentences:

            if (
                current_chunk_length +
                len(sentence)
            ) < chunk_size:

                current_chunk_sentences.append(
                    sentence
                )

                current_chunk_length += len(
                    sentence
                )

            else:

                chunk_text = " ".join(
                    current_chunk_sentences
                )

                section_aware_chunks.append({
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "text": chunk_text
                })

                chunk_id += 1

                current_chunk_sentences = [
                    sentence
                ]

                current_chunk_length = len(
                    sentence
                )

        # Flush remaining sentences
        if current_chunk_sentences:

            chunk_text = " ".join(
                current_chunk_sentences
            )

            section_aware_chunks.append({
                "chunk_id": chunk_id,
                "section": section_title,
                "text": chunk_text
            })

            chunk_id += 1

    return section_aware_chunks


# =========================================================
# EMBEDDING GENERATION
# =========================================================

def generate_normalized_embedding(text):

    embedding_vector = embedding_model.encode(
        text
    )

    normalized_embedding = (
        embedding_vector /
        np.linalg.norm(embedding_vector)
    )

    return normalized_embedding.tolist()


# =========================================================
# VECTOR DATABASE INDEXING
# =========================================================

def index_document_chunks(
    document_collection,
    section_aware_chunks
):

    for chunk_data in section_aware_chunks:

        chunk_text = chunk_data["text"]

        section_title = chunk_data["section"]

        chunk_embedding = (
            generate_normalized_embedding(
                chunk_text
            )
        )

        document_collection.add(

            documents=[chunk_text],

            embeddings=[chunk_embedding],

            metadatas=[{
                "section": section_title,
                "chunk_id": chunk_data[
                    "chunk_id"
                ]
            }],

            ids=[
                str(chunk_data["chunk_id"])
            ]
        )


# =========================================================
# VECTOR DATABASE SEARCH
# =========================================================

def search_vector_database(
    document_collection,
    query_embedding,
    top_k_results
):

    return document_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k_results
    )


# =========================================================
# FILTER RETRIEVED DOCUMENTS
# =========================================================

def filter_retrieved_documents(
    retrieved_documents,
    minimum_chunk_word_count
):

    return [

        document

        for document in retrieved_documents

        if len(document.split())
        > minimum_chunk_word_count
    ]


# =========================================================
# RERANK DOCUMENTS
# =========================================================

def rerank_documents(
    query_embedding,
    retrieved_documents,
    reranked_result_count
):

    reranked_documents = []

    for retrieved_document in retrieved_documents:

        document_embedding = (
            generate_normalized_embedding(
                retrieved_document
            )
        )

        similarity_score = np.dot(
            query_embedding,
            document_embedding
        )

        reranked_documents.append(
            (
                similarity_score,
                retrieved_document
            )
        )

    reranked_documents.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    # Prevent weak content from comprising main answer
    reranked_documents = [
    doc for score, doc in reranked_documents
    if score > SIMILARITY_THRESHOLD
    ]

    return [

        document

        for document
        in reranked_documents[
            :reranked_result_count
        ]
    ]


# =========================================================
# RETRIEVAL PIPELINE
# =========================================================

def retrieve_relevant_chunks(
    document_collection,
    user_query
):

    query_embedding = (
        generate_normalized_embedding(
            user_query
        )
    )

    search_results = (
        search_vector_database(
            document_collection,
            query_embedding,
            TOP_K_RETRIEVAL_RESULTS
        )
    )

    retrieved_documents = (
        search_results["documents"][0]
    )

    filtered_documents = (
        filter_retrieved_documents(
            retrieved_documents,
            MINIMUM_CHUNK_WORD_COUNT
        )
    )

    reranked_documents = (
        rerank_documents(
            query_embedding,
            filtered_documents,
            FINAL_RERANKED_RESULTS
        )
    )

    return reranked_documents


# =========================================================
# CONTEXT CONSTRUCTION
# =========================================================

def build_retrieved_context(
    retrieved_chunks
):

    return "\n".join(retrieved_chunks)


# =========================================================
# PROMPT CONSTRUCTION
# =========================================================

def build_prompt(
    retrieved_context,
    user_query
):

    return f"""
You are a strict retrieval-based assistant.

RULES:
- ONLY use context
- If answer is partially missing, say "not found"
- NEVER attempt interpretation
- NEVER explain if not explicitly stated

Context:
{retrieved_context}

Question:
{user_query}

Answer:
"""


# =========================================================
# RESPONSE GENERATION
# =========================================================

def generate_model_response(prompt):

    return ollama.chat(

        model=OLLAMA_MODEL_NAME,

        stream=True,

        options={

            "num_ctx": (
                OLLAMA_CONTEXT_WINDOW
            ),

            "num_predict": (
                OLLAMA_MAX_TOKENS
            ),

            "temperature": (
                OLLAMA_TEMPERATURE
            )
        },

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

# =========================================================
# QUERY NORMALISATION
# =========================================================

def preprocess_query(query):
    return query.lower().strip()

# =========================================================
# CHAT PIPELINE
# =========================================================

def answer_user_query(
    document_collection,
    user_query
):

    retrieved_chunks = (
        retrieve_relevant_chunks(
            document_collection,
            user_query
        )
    )

    # Fast fail if retrieval is weak
    if not retrieved_chunks:
        return "No relevant context found."

    retrieved_context = (
        build_retrieved_context(
            retrieved_chunks
        )
    )

    print("\n--- CONTEXT USED ---\n")

    print(retrieved_context)

    if (
        len(retrieved_context.split())
        < MINIMUM_CONTEXT_WORD_COUNT
    ):

        print(
            "\nNot enough relevant "
            "context to answer safely."
        )

        return

    prompt = build_prompt(
        retrieved_context,
        user_query
    )

    model_response = (
        generate_model_response(prompt)
    )

    print("\nAnswer:\n")

    for response_chunk in model_response:

        print(
            response_chunk["message"]["content"],
            end="",
            flush=True
        )


# =========================================================
# MAIN APPLICATION PIPELINE
# =========================================================

def main():

    document_collection = (
        initialize_vector_database()
    )

    raw_document_text = (
        load_document_text(
            DOCUMENT_PATH
        )
    )

    sanitized_document_text = (
        sanitize_document_text(
            raw_document_text
        )
    )

    structured_sections = (
        extract_document_sections(
            sanitized_document_text,
            SECTION_PATTERN
        )
    )

    section_aware_chunks = (
        build_section_aware_chunks(
            structured_sections,
            CHUNK_SIZE
        )
    )

    index_document_chunks(
        document_collection,
        section_aware_chunks
    )

    print(
        "\nDocument indexed successfully."
    )

    while True:

        user_query = input(
            "\nAsk a question: "
        )
        if user_query.lower() == "exit":
            break

        preprocessed_query = preprocess_query(user_query)
        answer_user_query(
            document_collection,
            preprocessed_query
        )

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()