import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os, hashlib, pickle

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

# Split text into chunks
def split_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create or Load Cached FAISS Vector Store
def create_vector_store(chunks, pdf_name="default"):
    os.makedirs("vector_store", exist_ok=True)

    # Generate unique hash for PDF
    pdf_hash = hashlib.md5(pdf_name.encode()).hexdigest()
    faiss_path = f"vector_store/{pdf_hash}_faiss"
    chunks_path = f"vector_store/{pdf_hash}_chunks.pkl"

    # If FAISS and chunks exist, load them
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        print("Loading cached FAISS index...")
        with open(chunks_path, "rb") as f:
            cached_chunks = pickle.load(f)
        if cached_chunks == chunks:  # Ensure chunks match (no new edits)
            return FAISS.load_local(faiss_path, 
                                    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                                    allow_dangerous_deserialization=True)

    # Else compute new embeddings
    print("Creating new FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(chunks, embeddings)

    # Save FAISS and chunks
    vectordb.save_local(faiss_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return vectordb
