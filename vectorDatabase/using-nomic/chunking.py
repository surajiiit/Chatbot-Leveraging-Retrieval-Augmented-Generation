import os
import pdfplumber
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# ----- CONFIGURATION -----
CHUNK_SIZE = 512  # Maximum tokens per chunk
CHUNK_OVERLAP = 50  # Overlapping tokens for better context
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load Nomic tokenizer
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# ----- LOAD TEXT FILES -----
def load_text_files(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    return documents

# ----- LOAD PDF FILES -----
def load_pdf_files(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                    documents.append(text)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    return documents

# ----- SEMANTIC CHUNKING FUNCTION -----
def semantic_chunking(text):
    """Splits text into semantically meaningful chunks while ensuring a max token limit of 512."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)

    final_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk, truncation=False)  # Ensure full tokenization
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk_tokens = tokens[i:i + CHUNK_SIZE]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            final_chunks.append(f"search_document: {chunk_text}")  # ✅ Best Practice Prefix
    
    return final_chunks

# ----- SAVE CHUNKS FUNCTION -----
def save_chunks(chunks, output_folder, filename_prefix):
    """Saves chunks efficiently with sequential numbering."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    existing_files = [f for f in os.listdir(output_folder) if f.startswith(filename_prefix)]
    start_index = len(existing_files) + 1

    for i, chunk in enumerate(chunks, start=start_index):
        chunk_file = os.path.join(output_folder, f"{filename_prefix}_chunk_{i}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)
        print(f"✅ Saved: {chunk_file}")

# 📂 Define input folders
pdf_folder = "/fab3/btech/2022/suraj.yadav22b/RAG/1. Vector/pdf"
txt_folder = "/fab3/btech/2022/suraj.yadav22b/RAG/1. Vector/finaltxt"
output_folder = "/fab3/btech/2022/suraj.yadav22b/RAG/1. Vector/nomic/nomicchunks"

# 📖 Load text and PDFs
txt_documents = load_text_files(txt_folder)
pdf_documents = load_pdf_files(pdf_folder)

# 🔥 Chunk and Save
for idx, doc in enumerate(txt_documents):
    chunks = semantic_chunking(doc)
    save_chunks(chunks, output_folder, f"txt_{idx+1}")

for idx, doc in enumerate(pdf_documents):
    chunks = semantic_chunking(doc)
    save_chunks(chunks, output_folder, f"pdf_{idx+1}")

print("✅ Semantic chunking for Nomic embeddings completed.")

