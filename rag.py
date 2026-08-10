import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_db():
    documents = []

    # PDFs load karo
    for file in os.listdir("uploads"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(
                os.path.join("uploads", file)
            )
            documents.extend(loader.load())

    # Agar PDF nahi mili
    if not documents:
        return False

    # Documents ko chunks mein divide karo
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS vector database
    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save database
    db.save_local("faiss_db")

    return True


if __name__ == "__main__":
    result = build_vector_db()

    if result:
        print("✅ Vector database created successfully!")
    else:
        print("❌ No PDF found in uploads folder.")