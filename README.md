# 📄 AI Career Consultant & Advanced RAG Explorer

An intelligent document analysis platform that goes beyond simple chatting. This tool performs **Strategic Gap Analysis** and **ATS Optimization** to help candidates align their resumes with specific job descriptions.

Built with **LangChain**, **OpenAI (GPT-4o-mini)**, and **Streamlit**.

## 🌟 Unique Value-Add Features

*   **🎯 Job Match & Gap Analysis:** Paste a Job Description to receive a detailed compatibility report, including a Match Score (%) and identified skill gaps.
*   **🛠️ ATS Optimization:** Specifically identifies missing technical keywords and semantic discrepancies that cause resumes to be filtered out by automated systems.
*   **⚙️ Dynamic Document Optimization:** Automatically adjusts Chunk Size and Search Depth ($k$) based on the document type (Resume vs. Legal vs. Technical) for maximum context accuracy.
*   **🔍 Smart Evidence UI:** Displays cleaned, relevant source citations in an easy-to-read format to verify AI claims.
*   **🛡️ Privacy-First:** Processes data in-memory and utilizes environment variables for API security. No data is stored persistently or used for training.

## 🛠️ Tech Stack
*   **Orchestration:** LangChain
*   **LLM:** OpenAI GPT-4o-mini (via ChatOpenAI)
*   **Embeddings:** text-embedding-3-small
*   **Vector Store:** ChromaDB (In-Memory)
*   **UI:** Streamlit

## ⚙️ Configuration
The system dynamically re-calibrates based on your selection:
- **Resume:** Large chunks to preserve professional experience context.
- **Legal/Book:** Smaller chunks with high overlap for deep cross-referencing.
- **Technical:** Broad search depth ($k=10$) for finding scattered code patterns.

## 💻 Installation & Usage

1. **Clone & Install:**
   ```bash
   git clone https://github.com
   cd pdf-rag-explorer
   pip install -r requirements.txt
