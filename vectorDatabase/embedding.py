from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import torch
import uuid
import os

# Ensure GPU is available
assert torch.cuda.is_available(), "GPU not available!"
torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = "cuda"

# Load the E5 model on GPU
model = SentenceTransformer("intfloat/e5-large", device=device)

# Load text chunks
chunk_dir = r"/fab3/btech/2022/suraj.yadav22b/RAG/1. Vector/chunks"
chunks = []
for filename in sorted(os.listdir(chunk_dir)):  # Sort for consistency
    filepath = os.path.join(chunk_dir, filename)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            chunks.append(file.read())

if not chunks:
    raise ValueError("No text chunks found!")

# Convert text chunks into embeddings
embeddings = model.encode(chunks, convert_to_numpy=True).tolist()

# Initialize Pinecone Client
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY is not set!")

pc = Pinecone(api_key=api_key)

index_name = "rag-index"

# Check if the index exists, create if not
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=len(embeddings[0]), metric="cosine")

index = pc.Index(name=index_name)

# Prepare vectors for upsert
vectors = [
    {"id": str(uuid.uuid4()), "values": embedding, "metadata": {"text": text[:500]}}  # Trim text
    for embedding, text in zip(embeddings, chunks)
]

# Upsert in smaller batches
batch_size = 50
for i in range(0, len(vectors), batch_size):
    batch = vectors[i : i + batch_size]
    index.upsert(batch)

print(f"Uploaded {len(vectors)} vectors to Pinecone.")
