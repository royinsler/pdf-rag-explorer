import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Load API Key
load_dotenv()

# Page configuration
st.set_page_config(page_title="Advanced PDF Chatbot", page_icon="📄")
st.title("📄 Advanced Document Analysis")

# --- 1. SIDEBAR CONFIGURATION (Defined first!) ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("API Key loaded from backend")
    
    st.divider()
    st.subheader("Optimization")
    doc_type = st.selectbox(
        "Select Document Type:",
        ["General", "Resume", "Legal/Book", "Technical/Code"]
    )

    # Define parameters BEFORE they are used
    if doc_type == "Resume":
        c_size, c_overlap, k_val = 1500, 100, 3
    elif doc_type == "Legal/Book":
        c_size, c_overlap, k_val = 800, 200, 7
    elif doc_type == "Technical/Code":
        c_size, c_overlap, k_val = 600, 50, 10
    else: # General
        c_size, c_overlap, k_val = 1000, 100, 5

    with st.expander("📊 Active Parameters", expanded=False):
        st.write(f"**Chunk Size:** {c_size}")
        st.write(f"**k:** {k_val}")

    st.divider()
    st.subheader("🎯 Job Match Analysis")
    job_description = st.text_area("Paste JD here:", height=150)
    analyze_clicked = st.button("Analyze Match")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.analysis_report = None
        st.rerun()

# --- 2. FILE UPLOADER & PROCESSING ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Process document
    with st.spinner("Processing..."):
        loader = PDFMinerLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
        splits = text_splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Initialize LLM and Chain
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': k_val}),
            return_source_documents=True
        )

    # --- 3. GAP ANALYSIS LOGIC ---
    if analyze_clicked and job_description:
        with st.spinner("🕵️ AI Consultant is analyzing your fit..."):
            gap_prompt = f"""
            First, identify the type of document provided. If it is a resume, perform a gap analysis against the JD as explained below. 
            If it's not a resume, kindly inform the user.
            Compare this resume with the Job Description: {job_description}.
            Return the output in this EXACT format:
            SCORE: [number between 0-100]
            REPORT: [Your detailed analysis, missing keywords, and 3 action items in markdown]
            """
            result = chain.invoke({"question": gap_prompt, "chat_history": []})
            full_res = result["answer"]
            
            # Extract score using simple parsing
            try:
                if "SCORE:" in full_res:
                    score_part = full_res.split("SCORE:")[1].split("\n")[0].strip()
                    # Clean any non-numeric characters like '%'
                    score_int = int(''.join(filter(str.isdigit, score_part)))
                    st.session_state.match_score = score_int
                    st.session_state.analysis_report = full_res.split("REPORT:")[1].strip()
            except:
                st.session_state.analysis_report = full_res

    # Display Analysis Report with Visuals
    if st.session_state.get("analysis_report"):
        st.divider()
        st.subheader("📊 Strategic Gap Analysis")
        
        # Visual Progress Bar
        score = st.session_state.get("match_score", 0)
        cols = st.columns([1, 4])
        cols[0].metric("Match Score", f"{score}%")
        
        # Color logic for the bar
        bar_color = "green" if score > 75 else "orange" if score > 40 else "red"
        cols[1].progress(score / 100)
        
        st.markdown(st.session_state.analysis_report)
        st.divider()


    # --- 4. CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            res = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
            answer = res["answer"]
            st.markdown(answer)
            
            with st.expander("🔍 Evidence"):
                for doc in res.get("source_documents", []):
                    st.markdown(f"> {doc.page_content[:300]}...")

            st.session_state.chat_history.append((prompt, answer))
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF to start.")
