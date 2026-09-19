import os
import faiss
import pickle
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# 1. LOAD API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found.")
    print("Please check your .env file.")
    exit()


# ==========================================
# 2. CONNECT TO GEMINI
# ==========================================

client = genai.Client(api_key=api_key)

print("Gemini connected successfully!")


# ==========================================
# 3. LOAD FAISS DATABASE
# ==========================================

print("Loading legal database...")

index = faiss.read_index(
    "vectorstore/legal.index"
)


# ==========================================
# 4. LOAD CHUNKS
# ==========================================

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 6. ASK USER
# ==========================================

question = input(
    "\nAsk a legal question about your document: "
)


# ==========================================
# 7. EMBED USER QUESTION
# ==========================================

question_embedding = embedding_model.encode(
    [question]
)

question_embedding = np.array(
    question_embedding
).astype("float32")


# ==========================================
# 8. SEARCH FAISS
# ==========================================

distances, indices = index.search(
    question_embedding,
    5
)


# ==========================================
# 9. GET RELEVANT CHUNKS
# ==========================================

context = ""

sources = []

for i, index_number in enumerate(indices[0]):

    chunk = chunks[index_number]

    context += (
        f"\n--- SOURCE {i + 1} ---\n"
        f"Page: {chunk['page']}\n"
        f"Text:\n{chunk['text']}\n"
    )

    sources.append(
        chunk["page"]
    )


# ==========================================
# 10. CREATE PROMPT
# ==========================================

prompt = f"""
You are an intelligent document research assistant.

Answer the user's question ONLY using the
provided legal document context.

IMPORTANT RULES:

1. Do not invent legal information.
2. Do not use information that is not present
   in the provided context.
3. If the answer cannot be found in the context,
   clearly say that the provided document does
   not contain enough information.
4. Explain the answer in simple language.
5. Mention the relevant page number.
6. Do not present your response as legal advice.
7. Distinguish the document's contents from your
   own explanation.

LEGAL DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Provide a clear answer and cite the relevant
page number.
"""


# ==========================================
# 11. SEND TO GEMINI
# ==========================================

print("\nGenerating answer...")

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)

answer = interaction.output_text


# ==========================================
# 12. DISPLAY ANSWER
# ==========================================

print("\n")
print("=" * 60)
print("LEGAL RAG ASSISTANT")
print("=" * 60)

print("\nANSWER:\n")

print(answer)

print("\n")
print("=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)

for page in sorted(set(sources)):
    print(f"📄 contract.pdf — Page {page}")