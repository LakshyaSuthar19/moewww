from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np


# ==========================================
# 1. LOAD FAISS DATABASE
# ==========================================

print("Loading database...")

index = faiss.read_index(
    "vectorstore/legal.index"
)


# ==========================================
# 2. LOAD CHUNKS
# ==========================================

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 4. ASK QUESTION
# ==========================================

question = input("\nAsk a question about the document: ")


# ==========================================
# 5. CONVERT QUESTION TO EMBEDDING
# ==========================================

question_embedding = model.encode(
    [question]
)

question_embedding = np.array(
    question_embedding
).astype("float32")


# ==========================================
# 6. SEARCH FAISS
# ==========================================

distances, indices = index.search(
    question_embedding,
    3
)


# ==========================================
# 7. SHOW RESULTS
# ==========================================

print("\n==============================")
print("MOST RELEVANT INFORMATION")
print("==============================")

for i, index_number in enumerate(indices[0]):

    print("\nResult:", i + 1)

    print(
        "Page:",
        chunks[index_number]["page"]
    )

    print(
        "Distance:",
        distances[0][i]
    )

    print(
        "Text:"
    )

    print(
        chunks[index_number]["text"]
    )

    print("--------------------------------")