import torch
import pinecone
from transformers import AutoModel, AutoTokenizer

# ----- CONFIGURATION -----
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
PINECONE_API_KEY = "#"  # Replace with your Pinecone API key
PINECONE_ENV = "us-east-1"  # Replace with your Pinecone environment
INDEX_NAME = "rag-nomic"  # Replace with your Pinecone index name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ----- INITIALIZE MODEL -----
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(DEVICE)
model.eval()

# ----- CONNECT TO PINECONE -----
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ----- FUNCTION TO SEARCH -----
def search_pinecone(query, top_k=5):
    """Searches Pinecone for the top K similar documents."""
    query_text = f"search_query: {query}"  # Best practice: Add prefix
    inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    
    return results

# Example usage
query_text = "Who is prof ferdous"
search_results = search_pinecone(query_text, top_k=3)

# Print results
for i, match in enumerate(search_results['matches']):
    print(f"{i+1}. Score: {match['score']} - {match['metadata']}")
