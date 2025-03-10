import os
from pinecone import Pinecone

from sentence_transformers import SentenceTransformer

# Load the E5 model
model = SentenceTransformer("intfloat/e5-large")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Ensure API key is set in the environment

# Connect to Pinecone index
index_name = "rag-index"
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found in Pinecone!")

index = pc.Index(index_name)

# Query Pinecone
query = "tell about prof ferdous"
query_embedding = model.encode(query, convert_to_numpy=True)

# Search in Pinecone
results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)

# Print results
# Print formatted results
for match in results["matches"]:
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print("-" * 50)

