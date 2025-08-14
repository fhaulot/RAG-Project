import os
from dotenv import load_dotenv
from rag_system import RAGSystem
from tqdm import tqdm

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    document_folder = "data/texts"

    print("Initializing RAG system...")
    rag = RAGSystem(api_key, document_folder)

    print("\nHello! I am your GDPR assistant. Ask me anything about your documents. Type 'exit' to quit.")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == "exit":
            break

        # Retrieve top 5 chunks
        hits = rag.retrieve_relevant_chunks(user_question, n_results=5)

        print("\nTop hits (preview, source, score):")
        for h in hits:
            preview = h["text"][:100].replace("\n", " ")
            print(f"- #{h['rank']} | score={h['score']:.3f} | {h['source']} | {preview}...")

        # Generate answer
        answer = rag.generate_answer(user_question, hits)
        print("\nAnswer:\n", answer)

