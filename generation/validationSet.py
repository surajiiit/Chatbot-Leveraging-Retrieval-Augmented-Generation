import torch
import requests
from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer

# ----- CONFIGURATION -----
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"  # Nomic model for local embedding
PINECONE_API_KEY = "#"  # Replace with actual API key
INDEX_NAME = "#"  # Replace with your Pinecone index name
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔥 Using device: {DEVICE}")

# ----- INITIALIZE EMBEDDING MODEL -----
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(DEVICE)
model.eval()

# ----- CONNECT TO PINECONE -----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ----- SYSTEM PROMPT -----
SYSTEM_PROMPT = (
    "You are a chat assistant. Find relevant information in the document for the given query. "
    "If the fact is present, summarize the answer and return it. Do not add any other response in the output."
)

# ----- FUNCTION TO SEARCH PINECONE -----
def search_pinecone(query, top_k=5):
    """Retrieves top K relevant documents from Pinecone using local Nomic embeddings."""
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    return results

# ----- FUNCTION TO QUERY LLAMA -----
def query_llama(query):
    """Queries Llama 3.2 with retrieved documents."""
    search_results = search_pinecone(query)
    
    # Extract relevant documents
    relevant_docs = "\n".join([match["metadata"].get("text", "") for match in search_results.get("matches", [])])

    full_prompt = f"{SYSTEM_PROMPT}\n\nRelevant Information:\n{relevant_docs}\n\nQuery: {query}"
    
    payload = {
        "model": "llama3.2",  # Adjust if using a different model name
        "prompt": full_prompt,
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    
    if response.status_code == 200:
        return response.json().get("response", "No response received")
    else:
        return f"Error: {response.status_code} - {response.text}"

# ----- VALIDATION QUERIES -----
queries = [
    "can CSE students change their branch to ECE?",
    "How many holidays are allowed in a single semester?",
    "What is the average package of CSE students?",
    "How many credits are required to complete a degree in CSE/ECE?",
    "How many professors are in the CSE department?",
    "What is the maximum reimbursement allowed from the college?",
    "What are the salaries of professors?",
    "What is the use of feedback given to any faculty?",
    "How many people leave college on average?",
    "Who are some famous alumni from the college?",
    "How many students can the hostels accommodate at max?",
    "How is college life? In terms of clubs and societies?",
    "What incentives does the college offer to entrepreneurs and startups?",
    "What are the major companies that visit college for hiring?",
    "Are there any anti-ragging policies?",
    "What is the mandatory attendance criteria followed here?",
]

# ----- RUN VALIDATION QUERIES -----
if __name__ == "__main__":
    results = {}

    for query in queries:
        print(f"\n🔎 Query: {query}")
        answer = query_llama(query)
        print(f"📝 Response: {answer}\n")
        results[query] = answer

    # Save results for manual verification
    import json
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Validation completed. Results saved to 'validation_results.json'.")
