import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load API Key from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with Your Documents")

# File uploader UI
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Verify the API key exists in the backend
    if not os.getenv("OPENAI_API_KEY"):
        st.error("API Key missing! Please set it in your .env file.")
        st.stop()

    # 1. Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 2. Initialize RAG components with a loading spinner
    with st.spinner("Analyzing document..."):
        # Load and split the document
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Create vector store in memory
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o-mini"),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
    
    st.success("Ready! Ask me anything about the file.")

    # 3. Chat history management (using Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Chat input handling
    if prompt := st.chat_input("What is this document about?"):
        # Add user message to history and UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using the RAG chain
        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            st.markdown(answer)
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a PDF file to begin.")
