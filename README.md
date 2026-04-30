# ⚡ InsightStream RAG

A high-performance **Retrieval-Augmented Generation (RAG)** application designed for accelerated career intelligence and document consultation. By leveraging asynchronous orchestration, **InsightStream RAG** achieves superior throughput compared to traditional sequential RAG pipelines.

## 🚀 Key Engineering Highlights

*   **⚡ Parallel Async Orchestration:** Utilizes `asyncio.gather` to fire multiple LLM chains simultaneously. This allows the "Strategic Gap Analysis" to perform deep matching and keyword extraction in parallel, reducing user latency by ~50%.
*   **🎯 Adaptive RAG Strategy:** Features a dynamic indexing engine that re-calibrates `Chunk Size`, `Overlap`, and `Retrieval Depth (k)` based on document intent (e.g., Resume vs. Legal).
*   **🛡️ Data Isolation & Privacy:** Implements isolated **ChromaDB** collections per session. All indexed data is ephemeral and can be explicitly incinerated via the UI to ensure 100% data privacy.
*   **📊 Live Performance Benchmarking:** Integrated telemetry to monitor and compare "Baseline" (Sequential) vs. "Async" (Parallel) execution times, providing transparent system metrics.

## 🛠️ Tech Stack

*   **Orchestration:** [LangChain](https://langchain.com) (Async API)
*   **LLM:** OpenAI GPT-4o-mini
*   **Vector DB:** [ChromaDB](https://trychroma.com)
*   **Frontend:** [Streamlit](https://streamlit.io)

## ⚙️ Performance Benchmarks (Typical)


| Operation | Strategy | Latency |
| :--- | :--- | :--- |
| **Simple Chat** | Sequential | ~7.8s |
| **Dual Gap Analysis** | **Parallel Async** | **~7.8s** |

*Insight: The system performs twice the computational work in the same time window as a single query.*

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd insightstream-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file:
   ```text
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Launch:**
   ```bash
   streamlit run app.py
   ```

---
Built by **Roy Insler** | [LinkedIn](www.linkedin.com/in/roy-insler-3a8042120)
