# RAG Project

In this repository, we'll implement a Retrieval-Augmented Generation (RAG) system using Google Generative AI. The system will be able to answer questions based on a set of documents by retrieving relevant information and generating coherent responses. The main components of the system include:

'''
RAG Project
├─ instructions.md
├─ README.md
├─ .gitignore
├─ .env
├─ .venv
├─ rag_launcher.py
├─ rag_system.py
├─ requirements.txt
└─ data
├─ previous_code
└─ texts


## Environment

Based on previous exercises, we reused some script previously used with the same architecture and structure. This allowed us to quickly adapt and implement the RAG system without starting from scratch. We used those librairies in a __ 3.12.10 Python environment __

- google.generativeai
- numpy
- python-dotenv
- scikit-learn
- textwrap
- faiss-cpu

So be sure that those librairies are well set on your machine. We also used the Google Generative AI API for the core functionality of the RAG system. The API key is required to authenticate requests and access the model's capabilities. As it is personal, ours is in a gitignore file called .env but you can create your own API Key on this link : https://aistudio.google.com/prompts/new_chat
Be careful not to expose your API key in public repositories! You should write it this way : GOOGLE_API_KEY=your_api_key_here (no ", (), < >, etc.)

## Usage 

To keep a clear and structured code, we choose to use OOP to get a modular design. This allows us to separate different concerns and functionalities into distinct classes and methods, making the codebase easier to maintain and extend. The OOP is in the RAG_System file. You can then use the Launching-RAG.py script to run the system. 
Allow it time to generate the documents embedding. Then, it will propose you to type your question. It will show you the 5 'top hits' to answer your question, this way you can find the source. Then it will answer your question. If you are out of context, it will help you to get in the context but won't answer if he is not able to find it. 




