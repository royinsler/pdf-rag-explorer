"""
Minimal standalone RAG script (CLI, no UI). Kept as a quick sanity-check
harness separate from the Streamlit app in app.py, which is the real entry
point for this project.
"""
import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def run_rag_system(file_path: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    print("Loading document...")
    loader = PDFMinerLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    if not texts:
        print("No readable text could be extracted from this PDF "
              "(it may be scanned/image-only or password-protected).")
        sys.exit(1)

    print("Creating vector database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(texts, embeddings)

    llm = ChatOpenAI(model_name="gpt-4o-mini")
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the following context:\n\n{context}"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    query = "What are the main points of this document?"
    print(f"\nQuestion: {query}")
    response = qa_chain.invoke({"input": query})

    print(f"\nAnswer: {response['answer']}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "my_document.pdf"
    run_rag_system(path)
