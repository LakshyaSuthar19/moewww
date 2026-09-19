import fitz
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "documents/contract.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text() + "\n"

print("PDF text extracted!")
print("Total characters:", len(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print("\n==============================")
    print("CHUNK:", i + 1)
    print("==============================")
    print(chunk)