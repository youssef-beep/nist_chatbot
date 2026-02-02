# NIST CSF 2.0 Incident Response Assistant

A specialized RAG-based AI system designed to answer questions strictly based on the **NIST SP 800-61r3** publication. This project demonstrates three core AI Engineering pillars: **RAG**, **Prompt Engineering (Routing)**, and **Synthetic Data Generation** for fine-tuning.

## 🚀 Features
1. **RAG (Retrieval-Augmented Generation):** Uses a local ChromaDB vector store and `all-MiniLM-L6-v2` embeddings to retrieve context from the PDF.
2. **JSON Routing:** A prompt-engineered gatekeeper that decides if a query is on-topic (RAG) or requires clarification.
3. **Synthetic Data Pipeline:** Automatically generates a "Question/Context/Answer" dataset from the PDF for potential fine-tuning.
4. **Local execution:** Runs entirely locally using **Ollama (Llama 3.2)** for privacy and zero cost.

## 🛠️ Tech Stack
- **LLM:** Llama 3.2 (via Ollama)
- **Framework:** LangChain
- **Vector DB:** ChromaDB
- **UI:** Streamlit
- **Embeddings:** HuggingFace (Sentence-Transformers)

## 📦 Setup Instructions

1. **Install Ollama:**
   Download and install from [ollama.com](https://ollama.com).
   Pull the model: `ollama pull llama3.2`

2. **Clone and Install:**
   ```bash
   git clone <your-repo-url>
   cd nist_chatbot
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   pip install -r requirements.txt