import streamlit as st
from utils import extract_text_from_pdf, split_text, create_vector_store
from transformers import pipeline
from dotenv import load_dotenv
import os, shutil

# === Load Environment Variables ===
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# === Configure Streamlit Page ===
st.set_page_config(page_title="AI PDF Assistant", layout="centered")
st.title("📄 AI Research Assistant (Hugging Face Edition)")

# === Sidebar: Cache Clear Button ===
st.sidebar.title("⚙️ Settings")
if st.sidebar.button("🗑️ Clear Cache"):
    if os.path.exists("vector_store"):
        shutil.rmtree("vector_store")
        os.makedirs("vector_store", exist_ok=True)
        st.sidebar.success("✅ Cache cleared successfully!")
    else:
        st.sidebar.info("No cache to clear.")

# === Load Pipelines Once & Cache ===
@st.cache_resource
def load_pipelines():
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # Smaller & faster than bart-large-cnn
        tokenizer="sshleifer/distilbart-cnn-12-6"
    )
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"  # Faster than roberta-base-squad2
    )
    return summarizer, qa

summarizer, qa = load_pipelines()

# === File Upload ===
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_pdf)
        chunks = split_text(text)
        vectordb = create_vector_store(chunks, pdf_name=uploaded_pdf.name)
        st.success(f"✅ PDF processed into {len(chunks)} chunks.")

        # Auto-select k
        k = 3 if len(chunks) <= 10 else 5 if len(chunks) <= 30 else 10

    # === Summarize Document ===
    with st.spinner("Summarizing the document..."):
        summary = summarizer(text[:3000], max_length=200, min_length=50, do_sample=False)[0]["summary_text"]

    st.markdown("### 📌 Document Summary:")
    st.write(summary)
    st.info(f"Auto-selected top {k} chunks based on document size.")

    # === Question Answering ===
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Searching for answers..."):
            docs = vectordb.similarity_search(query, k=k)
            combined_text = "\n".join([doc.page_content for doc in docs])
            result = qa(question=query, context=combined_text)
            answer = result["answer"]

        st.markdown("### 🧠 Answer:")
        st.write(answer)
