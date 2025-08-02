# 🧠 AI Research Assistant (Hugging Face Version)

An intelligent PDF-based research assistant that **summarizes and answers questions** about academic papers, technical documents, invoices, and more.  
Built with **Streamlit**, **LangChain**, **FAISS**, and **Hugging Face Transformers**.


<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-LangChain-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM-HuggingFace-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/UI-Streamlit-orange?style=flat-square" />
</p>

---

## 📌 Features

- ✅ Upload a research paper (PDF)
- 🧠 AI generates a **summary** instantly
- 🔍 Ask context-aware questions and get **accurate answers**
- 🧬 Uses vector search with **Hugging Face Embeddings + FAISS**
- ⚙️ Modular code design: easy to extend or deploy

---

## 📂 Folder Structure

```yaml
ai-research-assistant-huggingface/
│
├── app.py # Streamlit app interface
├── utils.py # PDF parsing, chunking, vector store setup
├── requirements.txt # Project dependencies
├── .env # API key (excluded via .gitignore)
├── sample_papers/ # Example PDFs (optional)
├── vector_store/ # Temporary FAISS vector index
├── .streamlit/
│   └── secrets.toml  # Not tracked by Git
└── README.md
```
---
## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Miraclechijioke/ai-research-assistant-huggingface.git
cd ai-research-assistant-huggingface

```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Add Hugging Face token
If you want to use private models, create a .env file:

```ini
HF_TOKEN=your_huggingface_token_here
```
### 4. Run the app
```bash
streamlit run app.py
```


## 🔄 Differences from the OpenAI Version

| Feature                | OpenAI Version                               | Hugging Face Version                     |
|------------------------|----------------------------------------------|------------------------------------------|
| **Embeddings**         | `OpenAIEmbeddings` (API key required)       | `HuggingFaceEmbeddings` (free models)   |
| **Summarization & QA** | GPT-based via OpenAI API                    | Transformers pipeline (`bart`, `t5`, etc.) |
| **Cost**               | Requires paid API usage                     | Mostly free with local/public models    |
| **Security**           | API key must be stored securely             | No API key needed for public models     |
| **Flexibility**        | Limited to OpenAI's models                  | Can choose any Hugging Face model       |


## 🛠 Built With
Streamlit – Web UI

LangChain – LLM pipelines and vector store integration

FAISS – Vector similarity search

Hugging Face Transformers – Embeddings, summarization & QA

PyMuPDF – Extract text from PDFs

## 🛡️ Security Note
No OpenAI API key is required.
If you use a Hugging Face token, never expose it publicly.
Use .streamlit/secrets.toml or .env for security.

## 👨‍💻 Author
**Miracle Chijioke Alozie**

_Associate Data Scientist & Machine Learning Engineer_

[LinkedIn](https://linkedin.com/in/mchijioke) • [GitHub](https://github.com/Miraclechijioke) • [Portfolio](https://bit.ly/MC_DataScience)

### ⭐️ Star This Repo
If you find this useful, feel free to give it a ⭐️ to support the project!
