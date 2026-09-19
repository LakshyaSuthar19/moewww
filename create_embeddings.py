from sentence_transformers import SentenceTransformer

sentences = [
    "A contract can be terminated with 30 days notice.",
    "The agreement requires written notice.",
    "The parties must follow the termination clause."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding size:", len(embeddings[0]))

print("\nFirst embedding:")
print(embeddings[0])