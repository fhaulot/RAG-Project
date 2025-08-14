## 4️⃣ Mini RAG (Retrieval-Augmented Generation) System

**Objective:**  
Build a small chatbot that answers questions based only on a provided document set.

**Constraints:**  
- Must not allow the LLM to answer from general knowledge (only from provided docs).  
- Must use an embedding-based retrieval step before generation.  
- Dataset should have at least 10+ documents.

**Dataset Ideas:**
- Lecture notes (PDF → text)
- Wikipedia articles on a single topic
- Product manuals

**Steps:**
1. Prepare your document set (convert to `.txt` or `.md`).  
2. Use an embedding model (e.g., OpenAI `text-embedding-ada-002` or sentence-transformers) to vectorize documents.  
3. Store embeddings in a vector database (e.g., FAISS, ChromaDB).  
4. On a query:
- Convert question to embedding
- Retrieve top relevant docs
- Pass retrieved text to LLM with instruction:  
  ```
  Answer the question using only the provided context. 
  If the answer is not in the context, say "I don't know."
  ```
5. Test with 5+ queries.

**Resources:**
- [FAISS Documentation](https://faiss.ai)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Sentence Transformers](https://www.sbert.net/)


# Deliverables for All Projects
- **Notebook or Python script** with working code  
- **README** explaining setup, usage, and findings  
- **Short report** summarizing results and challenges

## Process

First we reuse the code made in the exercises of the class. This allows us to build upon existing functionality and maintain consistency across the project. We then adapt the code to fit the specific requirements of the Mini RAG system, ensuring that it meets all constraints and objectives outlined in the project description. We structured it with OOP allowing us to separate the launcher of the RAG and the system used to launch it. We reuse the same embeddings and generative model from google. We used the same key, knowing that she worked. 

We made those functions : 

- `load_documents`: Loads and preprocesses the document set.
- `embed_documents`: Uses an embedding model to vectorize the documents.
- `create_index`: Saves the document embeddings in a vector database.
- `retrieve_best_chunk`: Retrieves relevant documents based on a query embedding.
- `generate_answer`: Generates an answer using the retrieved documents and the LLM.

Then we made the launcher : 
Some documents where not take into account during the embedding precess so we had to manage their extension. In the begininng, we also saw that the RAG system couldn't handle anything so we reshape the size of the chunks and of the query prompt to allow some space in the answers. 
