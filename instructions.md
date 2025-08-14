# RAG-Project
Objective:
Build a small chatbot that answers questions based only on a provided document set.

Constraints:

Must not allow the LLM to answer from general knowledge (only from provided docs).
Must use an embedding-based retrieval step before generation.
Dataset should have at least 10+ documents.
Dataset Ideas:

Lecture notes (PDF → text)
Wikipedia articles on a single topic
Product manuals
Steps:

Prepare your document set (convert to .txt or .md).
Use an embedding model (e.g., OpenAI text-embedding-ada-002 or sentence-transformers) to vectorize documents.
Store embeddings in a vector database (e.g., FAISS, ChromaDB).
On a query:
Convert question to embedding
Retrieve top relevant docs
Pass retrieved text to LLM with instruction:
Answer the question using only the provided context. 
If the answer is not in the context, say "I don't know."
Test with 5+ queries.
Resources

FAISS Documentation
LangChain RAG Tutorial
Sentence Transformers
Deliverables for All Projects
Notebook or Python script with working code
README explaining setup, usage, and findings
Short report summarizing results and challenges
