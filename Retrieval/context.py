import torch
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

# ----- CONFIGURATION -----
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
PINECONE_API_KEY = "#"  # Replace with your actual API key
INDEX_NAME = "rag-nomic"
TOP_K = 5  # Number of top chunks to retrieve
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Retrieval running on device: {DEVICE}")

# ----- Load Model & Tokenizer -----
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(DEVICE)
model.eval()

# ----- Connect to Pinecone -----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ----- Search Function -----
def search_chunks(query: str, top_k: int = TOP_K):
    """Return top_k relevant chunks from Pinecone for a given query."""
    query_text = f"search_query: {query}"
    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)

    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()

    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    chunks = [match["metadata"]["text"] for match in results["matches"]]

    return chunks

# ----- Load Validation Set -----
def load_validation_queries(path="validationSet.txt"):
    with open(path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    return queries

# ----- Run Retrieval and Save to File -----
def main():
    queries = load_validation_queries()
    print(f"📄 Loaded {len(queries)} queries from validationSet.txt")

    with open("needtoVerify.txt", "w", encoding="utf-8") as out_file:
        for i, query in enumerate(queries):
            out_file.write(f"[{i+1}] 🔍 Query: {query}\n")
            print(f"\n[{i+1}] 🔍 Query: {query}")
            
            top_chunks = search_chunks(query)

            for j, chunk in enumerate(top_chunks, 1):
                chunk_clean = chunk.replace("\n", " ").strip()
                chunk_preview = chunk_clean[:300] + ("..." if len(chunk_clean) > 300 else "")
                
                out_file.write(f"  {j}. {chunk_preview}\n")
                print(f"  {j}. {chunk_preview}")

            out_file.write("\n")

    print("✅ Retrieval results saved to needtoVerify.txt")

if __name__ == "__main__":
    main()

