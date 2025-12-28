# RAG_Pipeline
This project implements a Retrieval-Augmented Generation (RAG) pipeline that answers questions strictly from a PDF document using LangChain, FAISS, and Hugging Face models. The system retrieves relevant document chunks and generates answers based only on the retrieved context.

✨ Features
PDF ingestion using LangChain,
Recursive text chunking,
Semantic search using FAISS,
Open-source LLM via Hugging Face,
Runnable-based RAG pipeline,
Plain-text output (JSON not enforced).

## Project Structure
├── factory_spec.pdf
├── rag.py
├── requirements.txt
└── README.md

⚙️ Run Instructions
- Install dependencies by executing "pip install -r requirements.txt"
- Set Hugging Face API Token or change the code if want to use closed-source LLM
- Then run the program by "python rag.py"
- Edit the question directly in the script like :
result = main_chain.invoke("Your question here")
print(result)

🧠 How It Works (Brief)
PDF is loaded using PyPDFLoader
Text is split into overlapping chunks
Chunks are embedded and stored in FAISS
Relevant chunks are retrieved via similarity search
Retrieved context is passed to the LLM
The LLM generates an answer based on the context
