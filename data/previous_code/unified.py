import os
import google.generativeai as genai
import textwrap
import numpy as np
import faiss
import re
from typing import List
from dotenv import load_dotenv

# Using the CPU version of FAISS for a straightforward installation
print("Using faiss-cpu.")

class RAGSystem:
    """
    A class to manage a Retrieval-Augmented Generation (RAG) system.
    This version uses the FAISS library for fast and efficient similarity search
    on document chunks. It is configured to use the CPU version of Faiss.
    """

    def __init__(self, api_key: str, document_folder: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the RAG system by setting up the API, model configurations,
        loading documents, and creating the FAISS index.

        Args:
            api_key (str): The API key for accessing Google's models.
            document_folder (str): The path to the folder containing the Markdown files.
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of characters to overlap between sub-chunks.
        """
        if not api_key:
            raise ValueError('The GOOGLE_API_KEY is missing. Please define it in your .env file.')

        # API configuration
        genai.configure(api_key=api_key)

        self.generation_model_name = 'gemini-2.5-flash-lite'
        self.embedding_model_name = "models/embedding-001"
        self.generation_model = genai.GenerativeModel(self.generation_model_name)
        
        # Internal document processing
        self.document_chunks = self._load_documents_from_folder(document_folder, chunk_size, chunk_overlap)
        self.index = None

        if not self.document_chunks:
            raise ValueError("The document list cannot be empty.")
            
        # Create the index right after initializing the system
        self.create_index()

    def _load_documents_from_folder(self, folder_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Loads all .md files from a specified folder and returns a list of text chunks.
        The content is split into chunks of a fixed size, with a configurable overlap.
        This approach is more robust than splitting on specific headers.
        
        Args:
            folder_path (str): The path to the folder containing the Markdown files.
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of characters to overlap between sub-chunks.
        
        Returns:
            list: A list of strings, where each string is a processed chunk of text.
        """
        documents = []
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' was not found.")

        for filename in os.listdir(folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    content = content.replace('\n', ' ').replace('  ', ' ').strip() # Nettoyage du texte

                    # Découpage du texte en chunks de taille fixe
                    start_index = 0
                    while start_index < len(content):
                        end_index = start_index + chunk_size
                        sub_chunk = content[start_index:end_index]
                        documents.append(sub_chunk.strip())
                        # Move the index forward with overlap
                        start_index += chunk_size - chunk_overlap
        return documents

    def create_index(self):
        """
        Generates embeddings for the document chunks and creates the FAISS index.
        This method is called automatically during initialization.
        """
        print("Generating embeddings for documents...")
        # Generate embeddings for all documents
        chunk_embeddings = self._get_embeddings(self.document_chunks)
        
        # Ensure embeddings are in float32 format, as required by FAISS
        chunk_embeddings = np.array(chunk_embeddings).astype('float32')
        
        # Create and index the embeddings with FAISS
        dimension = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(chunk_embeddings)

        print(f"RAG system initialized with FAISS (CPU). {self.index.ntotal} documents have been indexed.")

    def _get_embeddings(self, content_list: list):
        embeddings = []
        for text in content_list:
            try:
                result = genai.embed_content(
                    model=self.embedding_model_name,
                    content=text
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Erreur embedding pour un chunk: {e}")
        return embeddings

    def retrieve_relevant_chunks(self, user_question: str, n_results: int = 3):
        """
        Retrieves the most relevant document chunks for a given question
        using the FAISS index.
        """
        if self.index is None:
            raise RuntimeError("FAISS index has not been created.")
            
        try:
            # Create the embedding for the user's question and convert to float32
            prompt_embedding = np.array(self._get_embeddings([user_question])).astype('float32')

            # Search the FAISS index
            distances, indices = self.index.search(prompt_embedding, n_results)

            # Retrieve the text of the relevant chunks using the indices
            relevant_chunks_text = [self.document_chunks[i] for i in indices[0]]
            return relevant_chunks_text
        except Exception as e:
            print(f"An error occurred while retrieving chunks: {e}")
            return []

    def generate_answer(self, user_question: str, context: list):
        """
        Generates a response to the user's question using the provided context.
        """
        if not context:
            return "I cannot answer the question based on the information provided."

        wrapped_context = "\n---\n".join(context)
        final_prompt = textwrap.dedent(f"""
        You are a GDPR assistant. 
        Answer in English, even if the question or context is in another language. 
        If the question is outside the provided context, help the user reformulate it so that it matches the context. 
        Then, answer based only on the context provided below. 
        If the answer is not in the context, clearly say so.


        Context:
        ---
        {wrapped_context}
        ---

        Question: {user_question}
        """)

        try:
            response = self.generation_model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while generating the response: {e}"
        
if __name__ == "__main__":
    # 1. Configuration
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    document_folder = 'data/texts'
    chunk_size = 500  # The size of each text chunk (in characters)
    chunk_overlap = 50 # The number of characters to overlap between chunks

    # 2. RAG system initialization
    try:
        rag_system = RAGSystem(api_key, document_folder, chunk_size, chunk_overlap)

        # 3. Chatbot simulation
        print("\nHello! I am a RAG chatbot. Ask me a question about the provided documents. (Type 'exit' to quit)")
        while True:
            user_input = input("\nYour question: ")
            if user_input.lower() == 'exit':
                break

            # 4. Retrieval of relevant chunks
            relevant_context = rag_system.retrieve_relevant_chunks(user_input)

            # 5. Generation and display of the answer
            answer = rag_system.generate_answer(user_input, relevant_context)
            print("\nRéponse: ", answer)
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
