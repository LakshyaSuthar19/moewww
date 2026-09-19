import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


# ==========================================
# 1. PDF LOCATION
# ==========================================

pdf_path = "documents/contract.pdf"


# ==========================================
# 2. READ PDF
# ==========================================

print("Reading PDF...")

doc = fitz.open(pdf_path)

pages = []

for page_number, page in enumerate(doc):
    text = page.get_text()

    pages.append({
        "page": page_number + 1,
        "text": text
    })

print("PDF loaded successfully!")
print("Total pages:", len(pages))


# ==========================================
# 3. CREATE CHUNKS
# ==========================================

print("\nCreating chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = []

for page in pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append({
            "text": chunk,
            "page": page["page"]
        })

print("Total chunks:", len(chunks))


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

print("\nCreating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings created!")
print("Embedding shape:", embeddings.shape)


# ==========================================
# 6. CREATE FAISS DATABASE
# ==========================================

print("\nCreating FAISS database...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS database created!")
print("Vectors stored:", index.ntotal)


# ==========================================
# 7. CREATE VECTORSTORE FOLDER
# ==========================================

os.makedirs("vectorstore", exist_ok=True)


# ==========================================
# 8. SAVE FAISS DATABASE
# ==========================================

faiss.write_index(
    index,
    "vectorstore/legal.index"
)


# ==========================================
# 9. SAVE CHUNK INFORMATION
# ==========================================

with open(
    "vectorstore/chunks.pkl",
    "wb"
) as f:

    pickle.dump(chunks, f)


print("\n================================")
print("SUCCESS!")
print("================================")

print("FAISS database saved.")
print("Location: vectorstore/legal.index")

print("Chunk information saved.")
print("Location: vectorstore/chunks.pkl")