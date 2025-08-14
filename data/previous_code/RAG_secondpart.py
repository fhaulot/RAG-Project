#Your code here

from sklearn.metrics.pairwise import cosine_similarity

user_prompt = "What are the rules for delay"

chunk_embeddings = genai.embed_content(
    model="models/embedding-001",
    content=chunks
)['embedding']

# Get the prompt embedding
prompt_embedding = genai.embed_content(
    model="models/embedding-001",
    content=user_prompt
)['embedding']

# Reshape for scikit-learn
prompt_vector = np.array(prompt_embedding).reshape(1, -1)
chunk_vectors = np.array(chunk_embeddings)

# Compute similarities for all chunks
similarities = cosine_similarity(prompt_vector, chunk_vectors)[0]

# Filter chunks based on a similarity threshold
required_similarity = 0.7
best_chunks_indices = np.where(similarities > required_similarity)[0]

# Retrieve the actual text of the best chunks
best_chunks_text = [chunks[i] for i in best_chunks_indices]

final_prompt = f"""
Based ONLY on the following context, answer the user's question.
If the answer is not in the context, say "I cannot answer the question based on the provided information."

Context:
---
{context}
---

Question: {user_prompt}
"""

# Get the final answer from Gemini
try:
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(final_prompt)
    print(response.text)
except Exception as e:
    print(f"Error occurred: {e}")