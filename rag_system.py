import os
import google.generativeai as genai
import textwrap
import numpy as np
import faiss
from typing import List
from tqdm import tqdm

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system with FAISS for similarity search
    and Google's Gemini model for text generation.
    """

    def __init__(self, api_key: str, document_folder: str, chunk_size: int = 500, chunk_overlap: int = 50):
        if not api_key:
            raise ValueError("The GOOGLE_API_KEY is missing.")
        genai.configure(api_key=api_key)

        self.generation_model_name = "gemini-2.5-flash-lite"
        self.embedding_model_name = "models/embedding-001"
        self.generation_model = genai.GenerativeModel(self.generation_model_name)

        # Load and chunk documents
        self.document_chunks, self.chunk_sources = self._load_documents_from_folder(
            document_folder, chunk_size, chunk_overlap
        )

        if not self.document_chunks:
            raise ValueError("No documents found in folder.")

        # Build FAISS index
        self.index = None
        self.create_index()

    def _load_documents_from_folder(self, folder_path: str, chunk_size: int, chunk_overlap: int):
        chunks = []
        sources = {}
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' not found.")

        for filename in os.listdir(folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().replace("\n", " ").replace("  ", " ").strip()
                    start = 0
                    while start < len(content):
                        end = start + chunk_size
                        chunk = content[start:end].strip()
                        chunks.append(chunk)
                        sources[len(chunks)-1] = filename
                        start += chunk_size - chunk_overlap
        return chunks, sources

    def _get_embeddings(self, texts: List[str]):
        """
        Generate embeddings using Google's embedding model.
        """
        try:
            # Supports batching if needed
            embeddings = genai.embed_content(model=self.embedding_model_name, content=texts)["embedding"]
            return np.array(embeddings).astype("float32")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.zeros((len(texts), 768), dtype="float32")  # fallback

    def create_index(self):
        """
        Build FAISS index from document embeddings.
        """
        print("Generating embeddings for documents...")
        embeddings = self._get_embeddings(self.document_chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"FAISS index created with {self.index.ntotal} chunks.")

    def retrieve_relevant_chunks(self, user_question: str, n_results: int = 3):
        """
        Return top-n relevant chunks with metadata: text, rank, score, source.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built.")

        q_emb = self._get_embeddings([user_question])
        distances, indices = self.index.search(q_emb, n_results)

        hits = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            hits.append({
                "text": self.document_chunks[idx],
                "rank": rank,
                "score": float(dist),
                "source": self.chunk_sources.get(idx, "unknown")
            })
        return hits

    def generate_answer(self, user_question: str, context: List[dict]):
        """
        Generate answer using context chunks. Responds in English.
        """
        if not context:
            return "I cannot answer the question based on the information provided."

        wrapped_context = "\n---\n".join([c["text"] for c in context])
        prompt = textwrap.dedent(f"""
        You are a GDPR assistant.
        Answer in English, even if the question or context is in another language.
        If the question is outside the provided context, help the user reformulate it so that it matches the context.
        Then, answer based only on the provided context.
        If the answer is not in the context, clearly say so.

        Context:
        ---
        {wrapped_context}
        ---

        Question: {user_question}
        """)

        try:
            response = self.generation_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"