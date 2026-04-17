# 📄 Advanced PDF Chatbot with RAG & Dynamic Optimization

An intelligent document assistant that allows you to chat with your PDF files. Built with **LangChain**, **OpenAI (GPT-4o-mini)**, and **Streamlit**.

## 🚀 Key Features
*   **Dynamic Document Optimization:** Tailor the AI's reading strategy based on document type (Resume, Legal, Technical, etc.).
*   **Contextual Memory:** Full conversation history support, allowing for natural follow-up questions.
*   **Smart Source Citations:** View exactly which parts of the document were used to generate each answer, with a clean UI.
*   **Session Management:** "Clear Chat" functionality to reset context without refreshing the app.
*   **Backend Security:** Secure API key handling via environment variables (.env).

## 🛠️ Tech Stack
*   **Framework:** LangChain
*   **LLM:** OpenAI GPT-4o-mini
*   **Embeddings:** text-embedding-3-small
*   **Vector Store:** ChromaDB
*   **Frontend:** Streamlit

## ⚙️ Configuration Parameters
The app dynamically adjusts these parameters based on your selection in the sidebar:
- **Chunk Size:** Optimizes the length of text segments for better context.
- **Overlap:** Ensures no information is lost between chunks.
- **Search Depth (k):** Controls how many document segments the AI reviews before answering.

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd pdf-rag-explorer
