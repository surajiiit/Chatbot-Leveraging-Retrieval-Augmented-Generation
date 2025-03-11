import os
import pdfplumber
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to read TXT files
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

# Function to read PDF files
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

# GPU-Accelerated function to chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    words_tensor = torch.tensor([ord(c) for c in " ".join(words)], dtype=torch.int64).to(device)
  # Convert text to tensor
    
    chunk_tensors = [words_tensor[i:i + chunk_size] for i in range(0, len(words_tensor), chunk_size)]
    chunk_texts = ["".join([chr(c.item()) for c in chunk]) for chunk in chunk_tensors]
    
    return chunk_texts

# Function to save chunks without overwriting previous files
def save_chunks(chunks, output_folder, filename_prefix):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find the last saved chunk number
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(filename_prefix)]
    start_index = len(existing_files) + 1
    
    for i, chunk in enumerate(chunks, start=start_index):
        chunk_file = os.path.join(output_folder, f"{filename_prefix}_chunk_{i}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)
        print(f"Saved: {chunk_file}")

# 📂 Define input folders
pdf_folder = "Enter_Pdf_Folder_path"
txt_folder = "Enter_txt_folder_path"
output_folder = "Output_folder"

# 📖 Load text and PDFs
txt_documents = load_text_files(txt_folder)
pdf_documents = load_pdf_files(pdf_folder)

# 🔥 Chunk and Save (GPU Accelerated)
for idx, doc in enumerate(txt_documents):
    chunks = chunk_text(doc)
    save_chunks(chunks, output_folder, f"txt_{idx+1}")

for idx, doc in enumerate(pdf_documents):
    chunks = chunk_text(doc)
    save_chunks(chunks, output_folder, f"pdf_{idx+1}")

print("✅ GPU-accelerated chunking completed without overwriting existing files.")
