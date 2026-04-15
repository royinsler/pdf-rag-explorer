import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Set up the API Key
load_dotenv()
def run_rag_system(file_path):
    # 2. Load the document (PDF in this case)
    print("Loading document...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 3. Split the text into small chunks
    # This ensures the information fits within the model's context window
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 4. Create Embeddings and store them in a Vector Database (ChromaDB)
    # This converts text into mathematical vectors for semantic search
    print("Creating vector database...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)

    # 5. Set up the Retrieval QA Chain
    # The model will search the database first and then generate an answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # 6. Run a sample query
    query = "What are the main points of this document?"
    print(f"\nQuestion: {query}")
    response = qa_chain.invoke(query)
    
    print(f"\nAnswer: {response['result']}")

if __name__ == "__main__":
    # Replace with the path to an actual PDF file on your machine
    run_rag_system("my_document.pdf")
