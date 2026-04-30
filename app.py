import streamlit as st
import os
import tempfile
import asyncio
import re
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Load API keys from environment
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream RAG", 
    page_icon="⚡", 
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
# Ensures all persistent variables are ready for the UI logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "last_source_docs" not in st.session_state:
    st.session_state.last_source_docs = []
if "last_chat_duration" not in st.session_state:
    st.session_state.last_chat_duration = None
if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None

# --- 1. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚡ InsightStream RAG")
    st.markdown("*High-Performance Career Intelligence*")
    st.markdown("---")
    
    # Live Performance Dashboard
    st.subheader("📊 Performance Metrics")
    if st.session_state.last_chat_duration:
        st.write(f"⏱️ **Baseline (Single):** {st.session_state.last_chat_duration:.2f}s")
    if st.session_state.analysis_time:
        st.write(f"🚀 **Async (Parallel):** {st.session_state.analysis_time:.2f}s")
    
    st.divider()
    st.subheader("🛠️ System Specs")
    st.write("- **Engine:** Async RAG Pipeline")
    st.write("- **Orchestration:** asyncio.gather")
    st.write("- **Storage:** ChromaDB (Isolated)")
    
    st.markdown("---")
    st.subheader("👤 Developer")
    st.markdown("[LinkedIn Profile](https://linkedin.com)")
    st.markdown("[GitHub Repository](https://github.com)")
    st.info("Built with LangChain & OpenAI GPT-4o-mini")

# --- 2. MAIN INTERFACE ---
st.title("⚡ InsightStream RAG")
st.markdown("Accelerated document intelligence with adaptive strategies and parallel LLM orchestration.")

col_file, col_opt = st.columns(2)

with col_file:
    uploaded_file = st.file_uploader("Upload Document (PDF)", type="pdf")

with col_opt:
    doc_type = st.selectbox(
        "Optimization Strategy:",
        ["General", "Resume", "Legal/Book", "Technical/Code"]
    )
    
    # Adaptive chunking/retrieval logic
    if doc_type == "Resume":
        c_size, c_overlap, k_val = 1500, 100, 3
    elif doc_type == "Legal/Book":
        c_size, c_overlap, k_val = 800, 200, 7
    elif doc_type == "Technical/Code":
        c_size, c_overlap, k_val = 600, 50, 10
    else:
        c_size, c_overlap, k_val = 1000, 100, 5
    st.caption(f"Strategy: {doc_type} | Chunk: {c_size} | k: {k_val}")

# --- 3. CORE RAG ENGINE ---
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Initializing InsightStream Engine..."):
        # Load and Index
        loader = PDFMinerLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
        splits = text_splitter.split_documents(docs)
        
        # Collection Isolation
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', uploaded_file.name)[:60]
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(),
            collection_name=safe_name
        )
        
        # Chain Initialization
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
            retriever=vectorstore.as_retriever(search_kwargs={'k': k_val}),
            return_source_documents=True
        )

    # --- 4. FEATURE TABS ---
    st.divider()
    tab_chat, tab_analysis = st.tabs(["💬 Interactive Consultant", "🎯 Strategic Analysis"])

    # Parallel Execution Logic
    async def run_parallel_analysis(jd):
        p1 = f"Analyze this resume against the JD: {jd}. Provide Match Score % and 3 key Action Items."
        p2 = f"List the top 5 missing technical keywords from this resume based on the JD: {jd}."
        
        task1 = st.session_state.chain.ainvoke({"question": p1, "chat_history": []})
        task2 = st.session_state.chain.ainvoke({"question": p2, "chat_history": []})
        
        results = await asyncio.gather(task1, task2)
        return results[0]["answer"], results[1]["answer"]

    # TAB 1: INTERACTIVE CHAT
    with tab_chat:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_chat_duration = None
            st.rerun()

        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"): st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)
                if st.session_state.last_source_docs:
                    with st.expander("🔍 View Context Evidence"):
                        for doc in st.session_state.last_source_docs:
                            st.markdown(f"> {doc.page_content[:400]}...")

        if prompt := st.chat_input("Consult the document..."):
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                start = time.time()
                res = st.session_state.chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                st.session_state.last_chat_duration = time.time() - start
                
                answer = res["answer"]
                st.markdown(answer)
                st.session_state.last_source_docs = res.get("source_documents", [])
                st.session_state.chat_history.append((prompt, answer))
                st.rerun()

    # TAB 2: STRATEGIC ANALYSIS
    with tab_analysis:
        st.subheader("Parallel Skill-Gap Analysis")
        st.write("Benchmarking document against target requirements via concurrent LLM orchestration.")
        job_description = st.text_area("Target Job Description:", height=200)
        
        if st.button("Run High-Speed Analysis", use_container_width=True):
            if job_description:
                start_time = time.time()
                with st.spinner("Executing Parallel Chains..."):
                    match_res, keyword_res = asyncio.run(run_parallel_analysis(job_description))
                    st.session_state.analysis_time = time.time() - start_time
                    st.session_state.analysis_report = f"{match_res}\n\n### 🔑 Missing Technical Keywords:\n{keyword_res}"
                    st.rerun()
            else:
                st.warning("Please input a Job Description to proceed.")

        if st.session_state.analysis_report:
            st.success("✅ Analysis Complete")
            st.markdown(st.session_state.analysis_report)
            if st.button("Clear Report"):
                st.session_state.analysis_report = None
                st.rerun()

else:
    st.info("👋 Welcome to InsightStream RAG. Please upload a PDF to initialize the engine.")
