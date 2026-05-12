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

CHUNK_SIZE = 350

TOP_K_RETRIEVAL_RESULTS = 15

FINAL_RERANKED_RESULTS = 5

NEIGHBOR_CHUNK_RADIUS = 1

MINIMUM_CHUNK_WORD_COUNT = 10

MINIMUM_CONTEXT_WORD_COUNT = 30

SIMILARITY_THRESHOLD = 0.28

OLLAMA_MODEL_NAME = "phi3"

OLLAMA_TEMPERATURE = 0.1

OLLAMA_CONTEXT_WINDOW = 1024

OLLAMA_MAX_TOKENS = 120


# =========================================================
# QUERY EXPANSION RULES
# =========================================================

QUERY_EXPANSIONS = {

    "worth": "honor virtue character righteousness",

    "wealth": "rich poverty money possessions",

    "study": "torah wisdom learning knowledge",

    "good person": "virtue righteousness character",

    "anger": "patience temper self control",

    "wisdom": "understanding knowledge torah",

    "israel": "the jewish people",

    "zionism": "the land of israel nationalism"
}


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

    raw_document_text = re.sub(
        r'\[[^\]]*\]',
        '',
        raw_document_text
    )

    raw_document_text = re.sub(
        r'\(\d+\)',
        '',
        raw_document_text
    )

    raw_document_text = re.sub(
        r'To see more[^.]*\.',
        '',
        raw_document_text,
        flags=re.IGNORECASE
    )

    raw_document_text = re.sub(
        r'\s+',
        ' ',
        raw_document_text
    )

    return raw_document_text.strip()


# =========================================================
# SECTION EXTRACTION
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
# SENTENCE SPLITTING
# =========================================================

def split_text_into_sentences(text):

    return re.split(
        r'(?<=[.!?])\s+',
        text
    )


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

        section_sentences = (
            split_text_into_sentences(
                section_text
            )
        )

        current_chunk_sentences = []

        current_chunk_character_count = 0

        for sentence in section_sentences:

            sentence_length = len(sentence)

            projected_chunk_size = (
                current_chunk_character_count
                + sentence_length
            )

            if projected_chunk_size < chunk_size:

                current_chunk_sentences.append(
                    sentence
                )

                current_chunk_character_count += (
                    sentence_length
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

                current_chunk_character_count = (
                    sentence_length
                )

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
# DOCUMENT INDEXING
# =========================================================

def index_document_chunks(
    document_collection,
    section_aware_chunks
):

    chunk_texts = [

        chunk["text"]

        for chunk in section_aware_chunks
    ]

    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    # Batch each chunk embedding to save processing power
    for chunk_data, chunk_embedding in zip(
        section_aware_chunks,
        chunk_embeddings
    ):

        chunk_data["embedding"] = (
            chunk_embedding.tolist()
        )

        document_collection.add(

            documents=[
                chunk_data["text"]
            ],

            embeddings=[
                chunk_data["embedding"]
            ],

            metadatas=[{

                "chunk_id":
                chunk_data["chunk_id"],

                "section":
                chunk_data["section"]
            }],

            ids=[
                str(chunk_data["chunk_id"])
            ]
        )


# =========================================================
# QUERY EXPANSION
# =========================================================

def expand_user_query(user_query):

    expanded_query = user_query

    lower_query = user_query.lower()

    for keyword, expansion in (
        QUERY_EXPANSIONS.items()
    ):

        if keyword in lower_query:

            expanded_query += (
                " " + expansion
            )

    return expanded_query


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
# BUILD RETRIEVED CHUNK OBJECTS
# =========================================================

def build_retrieved_chunk_objects(
    search_results,
    chunk_lookup
):

    retrieved_documents = (
        search_results["documents"][0]
    )

    retrieved_metadatas = (
        search_results["metadatas"][0]
    )

    retrieved_chunk_objects = []

    # Batch embeddings and enrich retrieved chunks with cached embeddings
    for document_text, metadata in zip(
        retrieved_documents,
        retrieved_metadatas
    ):

        chunk_id = metadata["chunk_id"]

        original_chunk = (
            chunk_lookup[chunk_id]
        )

        retrieved_chunk_objects.append({

            "chunk_id": chunk_id,

            "section": metadata["section"],

            "text": document_text,

            "embedding":
            original_chunk["embedding"]
        })

    return retrieved_chunk_objects


# =========================================================
# FILTER RETRIEVED CHUNKS
# =========================================================

def filter_retrieved_chunks(
    retrieved_chunk_objects,
    minimum_chunk_word_count
):

    filtered_chunks = []

    for chunk_object in (
        retrieved_chunk_objects
    ):

        if (
            len(
                chunk_object["text"].split()
            )
            >= minimum_chunk_word_count
        ):

            filtered_chunks.append(
                chunk_object
            )

    return filtered_chunks


# =========================================================
# RERANK RETRIEVED CHUNKS
# =========================================================

def rerank_retrieved_chunks(
    query_embedding,
    retrieved_chunk_objects,
    reranked_result_count
):

    reranked_chunks = []

    for chunk_object in (
        retrieved_chunk_objects
    ):

        similarity_score = np.dot(

            query_embedding,

            chunk_object["embedding"]
        )

        if (
            similarity_score
            >= SIMILARITY_THRESHOLD
        ):

            chunk_object[
                "similarity_score"
            ] = similarity_score

            reranked_chunks.append(
                chunk_object
            )

    reranked_chunks.sort(

        reverse=True,

        key=lambda chunk: (
            chunk["similarity_score"]
        )
    )

    return reranked_chunks[
        :reranked_result_count
    ]

# =========================================================
# NEIGHBOR CHUNK EXPANSION
# =========================================================

def expand_neighbor_chunks(
    chunk_lookup,
    retrieved_chunk_objects,
    neighbor_radius
):

    expanded_chunks = []

    seen_chunk_ids = set()
    # Use dictionary lookups to speed up retrieval
    for retrieved_chunk in (
        retrieved_chunk_objects
    ):

        center_chunk_id = (
            retrieved_chunk["chunk_id"]
        )

        center_section = (
            retrieved_chunk["section"]
        )

        for offset in range(
            -neighbor_radius,
            neighbor_radius + 1
        ):

            target_chunk_id = (
                center_chunk_id + offset
            )

            matching_chunk = (
                chunk_lookup.get(
                    target_chunk_id
                )
            )

            if not matching_chunk:
                continue

            if (
                matching_chunk["section"]
                != center_section
            ):
                continue

            if (
                target_chunk_id
                in seen_chunk_ids
            ):
                continue

            expanded_chunks.append(
                matching_chunk
            )

            seen_chunk_ids.add(
                target_chunk_id
            )

    expanded_chunks.sort(

        key=lambda chunk: (
            chunk["chunk_id"]
        )
    )

    return expanded_chunks

# =========================================================
# RETRIEVAL PIPELINE
# =========================================================

def retrieve_relevant_chunks(
    document_collection,
    chunk_lookup,
    user_query
):

    expanded_query = (
        expand_user_query(
            user_query
        )
    )

    query_embedding = (
        generate_normalized_embedding(
            expanded_query
        )
    )

    search_results = (
        search_vector_database(

            document_collection,

            query_embedding,

            TOP_K_RETRIEVAL_RESULTS
        )
    )

    retrieved_chunk_objects = (
        build_retrieved_chunk_objects(

            search_results,

            chunk_lookup
        )
    )

    filtered_chunk_objects = (
        filter_retrieved_chunks(

            retrieved_chunk_objects,

            MINIMUM_CHUNK_WORD_COUNT
        )
    )

    reranked_chunk_objects = (
        rerank_retrieved_chunks(

            query_embedding,

            filtered_chunk_objects,

            FINAL_RERANKED_RESULTS
        )
    )

    expanded_chunk_objects = (
        expand_neighbor_chunks(

            chunk_lookup,

            reranked_chunk_objects,

            NEIGHBOR_CHUNK_RADIUS
        )
    )

    return expanded_chunk_objects


# =========================================================
# CONTEXT CONSTRUCTION
# =========================================================

def build_retrieved_context(
    retrieved_chunk_objects
):

    context_sections = []

    for chunk_object in (
        retrieved_chunk_objects
    ):

        formatted_chunk = (

            f"[{chunk_object['section']}]\n"

            f"{chunk_object['text']}"
        )

        context_sections.append(
            formatted_chunk
        )

    return "\n\n".join(
        context_sections
    )


# =========================================================
# PROMPT CONSTRUCTION
# =========================================================

def build_prompt(
    retrieved_context,
    user_query
):

    return f"""
You are a retrieval-based assistant.

STRICT RULES:
- ONLY answer using the provided context
- NEVER use outside knowledge
- NEVER infer missing information
- NEVER generalize
- NEVER summarize broadly
- ONLY state information explicitly written
- ONLY answer in complete full sentences.
- If the answer is missing, say:
"I don't know based on the provided context."

Context:
{retrieved_context}

Question:
{user_query}

Answer:
"""


# =========================================================
# MODEL RESPONSE GENERATION
# =========================================================

def generate_model_response(prompt):

    return ollama.chat(

        model=OLLAMA_MODEL_NAME,

        stream=True,

        options={

            "num_ctx":
            OLLAMA_CONTEXT_WINDOW,

            "num_predict":
            OLLAMA_MAX_TOKENS,

            "temperature":
            OLLAMA_TEMPERATURE
        },

        messages=[

            {
                "role": "user",
                "content": prompt
            }
        ]
    )

# =========================================================
# QUERY PREPROCESSING
# =========================================================

def preprocess_query(user_query):
    return user_query.lower().strip()

# =========================================================
# ANSWER USER QUERY
# =========================================================

def answer_user_query(
    document_collection,
    all_chunks,
    user_query
):

    retrieved_chunks = (
        retrieve_relevant_chunks(

            document_collection,

            all_chunks,

            user_query
        )
    )

    if not retrieved_chunks:

        print(
            "\nNo relevant context found."
        )

        return

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
            "\nNot enough context "
            "to answer safely."
        )

        return

    prompt = build_prompt(
        retrieved_context,
        user_query
    )

    model_response = (
        generate_model_response(
            prompt
        )
    )

    print("\nAnswer:\n")

    for response_chunk in (
        model_response
    ):

        print(

            response_chunk[
                "message"
            ]["content"],

            end="",

            flush=True
        )


# =========================================================
# MAIN APPLICATION
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

    chunk_lookup = {

        chunk["chunk_id"]: chunk

        for chunk in section_aware_chunks
    }

    print(
        "\nDocument indexed successfully."
    )

    while True:

        user_query = input(
            "\nAsk a question: "
        )

        if user_query.lower() == "exit":
            break

        preprocessed_query = (
            preprocess_query(
                user_query
            )
        )

        answer_user_query(

            document_collection,

            chunk_lookup,

            preprocessed_query
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()