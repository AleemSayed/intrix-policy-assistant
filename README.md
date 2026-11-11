Intrix Policy Assistant

An AI-powered company policy assistant that uses Retrieval-Augmented Generation (RAG) to answer employee questions based on company policy documents.

Project Overview

Intrix Policy Assistant allows employees to ask questions about company policies in natural language. It retrieves relevant information from policy documents and uses a Large Language Model (LLM) to generate contextual answers.

The application is designed to provide accurate, document-grounded responses and avoid generating unsupported information.

Features

* Natural-language question answering
* Retrieval-Augmented Generation (RAG) pipeline
* PDF document ingestion and text chunking
* Vector-based semantic search using ChromaDB
* Groq API integration for LLM responses
* Streamlit-based interactive interface
* Responses based on retrieved policy information

Technology Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Groq API
* Hugging Face embeddings
* PyPDF

Project Structure

Policy Assistant/
├── data/
│   └── policies/
│       ├── expense_policy.pdf
│       ├── it_security_policy.pdf
│       └── leave_policy.pdf
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── AI_USAGE.md
└── TEST_RESULTS.md

How It Works

1. Document Ingestion: Policy PDFs are loaded and their text is extracted.
2. Text Chunking: Extracted text is divided into smaller chunks.
3. Embeddings: Text chunks are converted into vector embeddings.
4. Vector Storage: Embeddings are stored in ChromaDB.
5. Retrieval: Relevant chunks are retrieved based on the employee’s question.
6. Response Generation: The Groq-powered LLM generates an answer using the retrieved context.

Installation and Setup

1. Clone the repository

git clone https://github.com/AleemSayed/intrix-policy-assistant.git
cd intrix-policy-assistant

2. Create a virtual environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Copy .env.example to .env and add your Groq API key using the variable name specified in .env.example.

Never upload your .env file or expose your API key.

5. Build the vector database

python ingest.py

6. Run the application

streamlit run app.py

Sample Questions

* What is the annual casual leave entitlement?
* What is the hotel reimbursement limit?
* Which expenses are not reimbursable?
* What are the IT security requirements?
* How should an employee report a security incident?

Limitations

* Answers depend on the information available in the supplied policy documents.
* The assistant may not answer questions when relevant information is missing from the documents.
* It does not replace official HR, finance, or IT guidance.

Project Documentation

* AI_USAGE.md – AI assistance and usage documentation.
* TEST_RESULTS.md – Application testing and sample results.

Author

Aleem Azhar Sayed

GitHub Repository