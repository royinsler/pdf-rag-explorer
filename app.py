import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

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
        
    
    st.success("Ready! Ask me anything about the file.")

    # 3. Chat history management (using Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # For the LLM memory

    # Display previous messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 4. Chat input handling
    if prompt := st.chat_input("Ask me anything about the document..."):
        # Add user message to UI and session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Initialize the Conversational Chain with memory support
            # This chain combines the chat history with the new question
            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-4o-mini"),
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            # Generate response by passing the current question and history
            result = chain.invoke({
                "question": prompt, 
                "chat_history": st.session_state.chat_history
            })
            
            answer = result["answer"]
            source_docs = result["source_documents"]

            # Display the answer
            st.markdown(answer)
            
            # Save the pair (question, answer) to the LLM's history
            st.session_state.chat_history.append((prompt, answer))
            
            # Save the assistant response to the UI history
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display source citations in an expandable section
            with st.expander("View Sources"):
                for i, doc in enumerate(source_docs):
                    page_num = doc.metadata.get("page", 0) + 1
                    st.write(f"**Source {i+1} (Page {page_num}):**")
                    st.info(doc.page_content[:200] + "...")


else:
    st.info("Please upload a PDF file to begin.")
