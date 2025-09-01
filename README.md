# AI Course RAG Chat Interface

This project is a **Retrieval-Augmented Generation (RAG)** chatbot using PDF documents as the knowledge base. Users can ask questions in natural language, and the system retrieves relevant context from PDF documents to answer.


- Load multiple PDFs from a folder
- Split documents into smaller chunks for better retrieval
- Use HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Vector search using (FAISS)
- Question-answering using a HuggingFace LLM (`google/flan-t5-large`)
- Gradio interface for interactive chat
- Shows retrieved sources alongside answers


