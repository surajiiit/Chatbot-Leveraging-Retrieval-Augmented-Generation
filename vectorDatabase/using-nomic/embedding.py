import os
import torch
from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer

# ----- CONFIGURATION -----
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
PINECONE_API_KEY = "#"  # Replace with your Pinecone API key
INDEX_NAME = "rag-nomic"  # Replace with your Pinecone index name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----- INITIALIZE NOMIC EMBEDDING MODEL -----
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(DEVICE)
model.eval()

# ----- CONNECT TO PINECONE -----
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, otherwise create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric='cosine'  # Change to 'euclidean' if needed
    )

index = pc.Index(INDEX_NAME)

# ----- FUNCTION TO LOAD CHUNKED TEXT -----
def load_chunks(chunk_folder):
    """Loads all chunked text files from the specified folder."""
    if not os.path.exists(chunk_folder):
        print(f"⚠️ Warning: Chunk folder '{chunk_folder}' not found. Exiting.")
        exit(1)
    
    chunks = []
    for file in sorted(os.listdir(chunk_folder)):  # Ensures consistent order
        if file.endswith(".txt"):
            file_path = os.path.join(chunk_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                chunks.append(f"search_document: {text}")  # Best practice: Add document prefix
    return chunks

# ----- FIXED FUNCTION TO GENERATE NOMIC EMBEDDINGS (BATCH PROCESSING) -----
def generate_embeddings(text_chunks, batch_size=8):
    """Generates embeddings in batches using Nomic model for efficiency."""
    if not text_chunks:
        print("Warning: No chunks to process")
        return []
        
    embeddings = []
    with torch.no_grad():  # Disable gradient calculation
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i : i + batch_size]
            try:
                # Debug info
                print(f"Processing batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}, size: {len(batch)}")
                
                # Properly handle tokenization with padding
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(DEVICE)
                
                # Get model output properly
                with torch.cuda.amp.autocast(enabled=True):  # Mixed precision to help with memory
                    output = model(**inputs)
                
                # The correct way to access embeddings depends on the model's output structure
                # For Nomic models, use this approach:
                batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().detach().numpy()
                embeddings.extend(batch_embeddings)
                
                # Free up CUDA memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Continue processing other batches
                torch.cuda.empty_cache()  # Clear GPU memory
                continue
    
    return embeddings

# ----- FUNCTION TO UPLOAD TO PINECONE -----
def upload_to_pinecone(text_chunks, embeddings):
    """Uploads embeddings to Pinecone index."""
    vectors = [(f"doc_{i}", embedding.tolist(), {"text": chunk}) for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))]
    
    # Upload in batches to avoid request size limits
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(batch)
        print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} to Pinecone")
    
    print(f"✅ {len(vectors)} embeddings uploaded to Pinecone.")

# 📂 Load chunked text
chunk_folder = "/content/extracted/nomicchunks"
chunks = load_chunks(chunk_folder)
print(f"Loaded {len(chunks)} chunks from {chunk_folder}")

# 🔥 Generate embeddings with batch processing
embeddings = generate_embeddings(chunks)

# Check if generation was successful
if embeddings and len(embeddings) > 0:
    # 📤 Upload to Pinecone
    upload_to_pinecone(chunks, embeddings)
    print("✅ Nomic embeddings generated and stored in Pinecone.")
else:
    print("❌ Failed to generate embeddings. Check the errors above.")

# ----- FUNCTION TO SEARCH -----
def search_pinecone(query, top_k=5):
    """Searches Pinecone for the top K similar documents."""
    query_text = f"search_query: {query}"  # Best practice: Add query prefix
    inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    
    return results
