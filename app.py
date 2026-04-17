import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load API Key from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with Your Documents")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("API Key loaded from backend")
    
    st.divider()
    st.subheader("Document Optimization")
    doc_type = st.selectbox(
        "Select Document Type:",
        ["General", "Resume", "Legal/Book", "Technical/Code"],
        help="Adjusts chunk size and search depth."
    )

    # Dynamic parameters based on selection
    if doc_type == "Resume":
        c_size, c_overlap, k_val = 1500, 100, 3
    elif doc_type == "Legal/Book":
        c_size, c_overlap, k_val = 800, 200, 7
    elif doc_type == "Technical/Code":
        c_size, c_overlap, k_val = 600, 50, 10
    else: # General
        c_size, c_overlap, k_val = 1000, 100, 5

    # --- NEW: Displaying the actual values ---
    with st.expander("📊 Active Parameters", expanded=True):
        st.write(f"**Chunk Size:** `{c_size}`")
        st.write(f"**Overlap:** `{c_overlap}`")
        st.write(f"**Search Depth (k):** `{k_val}`")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()



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
        loader = PDFMinerLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,       # Use the variable from sidebar
            chunk_overlap=c_overlap  # Use the variable from sidebar
        )
        splits = text_splitter.split_documents(docs)
        
        # Create vector store in memory
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
    
    st.success("Ready! Ask me anything about the file.")

     # 3. Chat history management
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Chat input handling
    if prompt := st.chat_input("Ask me anything about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # We'll use a simpler but effective retrieval method
            # Increase 'k' to 5 to give the model more context to work with
            base_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
            
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

            # Creating the chain with the base retriever again to avoid over-filtering
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=base_retriever,
                return_source_documents=True
            )

            # Generate response
            with st.spinner("Thinking..."):
                result = chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })

            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])

            # Display the answer
            st.markdown(answer)
            
                    # Display source citations in a cleaner, wrapped format
            if source_docs:
                with st.expander("🔍 Evidence from Document"):
                    seen_content = set()
                    for i, doc in enumerate(source_docs):
                        content = doc.page_content.replace('\n', ' ').strip()
                        
                        if content and content not in seen_content:
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**📍 Source {i+1} — Page {page_num}**")
                            
                            # Using > for Blockquote instead of ``` for Code Block
                            # This ensures the text wraps naturally to the next line
                            st.markdown(f"> {content}")
                            
                            st.divider() # Adds a nice thin line between sources
                            seen_content.add(content)

            else:
                st.info("No relevant context was found in the document for this answer.")

            # Update history
            st.session_state.chat_history.append((prompt, answer))
            st.session_state.messages.append({"role": "assistant", "content": answer})
                        

else:
    st.info("Please upload a PDF file to begin.")
