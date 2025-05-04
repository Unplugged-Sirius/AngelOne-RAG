# Angel One RAG Assistant 🤖

A Retrieval Augmented Generation (RAG) powered chatbot for Angel One customer support using Streamlit, FAISS, and HuggingFace models.

## 🌟 Features

- Interactive chat interface built with Streamlit
- FAISS vector database for efficient document retrieval
- Automatic model selection based on available hardware (GPU/CPU)
- Optimized for both high and low-resource environments
- Clean response formatting with minimal hallucinations

## 🛠️ Technical Stack

- **Frontend:** Streamlit
- **Embeddings:** HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store:** FAISS
- **LLM Models:**
  - **GPU:** TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
  - **CPU:** TinyLlama/TinyLlama-1.1B-Chat-v1.0

## 📋 Prerequisites

- Python 3.8+
- 16GB+ RAM recommended
- CUDA-capable GPU (optional)
- Linux/Unix environment

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/angel-one-rag.git
cd angel-one-rag
```

### 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 💾 Project Structure

```
angel-one-rag/
├── app.py                     # Main Streamlit application
├── embeddingsNStorings.py     # FAISS index creation script
├── data.json                  # Knowledge base
├── faiss_index/              # Vector store
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

## ⚙️ Setup

### 1. Create FAISS index

```bash
python embeddingsNStorings.py
```

### 2. Launch application

```bash
streamlit run app.py
```

## 🔧 Configuration

The application automatically configures itself based on your hardware:
- **GPU Available:** Uses Mistral-7B-Instruct-v0.2-GPTQ
- **CPU Only:** Falls back to TinyLlama-1.1B-Chat

## 🔍 Usage

1. Type your question in the input field
2. Click "Submit" or press Enter
3. View the AI-generated response

## ⚠️ Troubleshooting

### Common Issues

#### Out of Memory
- **Solution:** The app will automatically fall back to smaller models
- Close unnecessary applications

#### CUDA Errors
- Verify CUDA installation: `nvidia-smi`
- Update GPU drivers
- System will fall back to CPU if needed

#### Slow Responses
- Check internet connection
- Verify system resources
- First run downloads models (be patient)