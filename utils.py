import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os, hashlib, pickle

# Extract text from PDF (optionally specific pages)
def extract_text_from_pdf(pdf_file, pages=None):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except Exception as e:
        raise ValueError(f"Error reading PDF: {e}")

    text = ""
    try:
        if pages:  # e.g., pages=[0,1,2] for first 3 pages
            for p in pages:
                if p < len(doc):
                    text += doc[p].get_text()
        else:  # whole PDF
            for page in doc:
                text += page.get_text()
    except Exception as e:
        raise ValueError(f"Error extracting text: {e}")

    if not text.strip():
        raise ValueError("No text found in PDF.")
    return text

# Split text into chunks
def split_text(text):
    if not text.strip():
        raise ValueError("Empty text passed for splitting.")
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create or Load Cached FAISS Vector Store
def create_vector_store(chunks, pdf_name="default"):
    if not chunks:
        raise ValueError("No chunks to embed.")

    os.makedirs("vector_store", exist_ok=True)

    pdf_hash = hashlib.md5(pdf_name.encode()).hexdigest()
    faiss_path = f"vector_store/{pdf_hash}_faiss"
    chunks_path = f"vector_store/{pdf_hash}_chunks.pkl"

    try:
        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            print("Loading cached FAISS index...")
            with open(chunks_path, "rb") as f:
                cached_chunks = pickle.load(f)
            if cached_chunks == chunks:
                return FAISS.load_local(
                    faiss_path,
                    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                    allow_dangerous_deserialization=True
                )
    except Exception as e:
        print(f"Cache loading failed, recreating index: {e}")

    print("Creating new FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(chunks, embeddings)

    # Save cache safely
    try:
        vectordb.save_local(faiss_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return vectordb
