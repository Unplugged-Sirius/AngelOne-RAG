import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Step 1: Load your JSON data
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Convert to LangChain Document format
docs = [
    Document(
        page_content=item["content"],
        metadata=item["metadata"]
    )
    for item in data
]

# Step 3: Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create FAISS index
faiss_index = FAISS.from_documents(docs, embedding_model)

# Step 5: Save index locally
faiss_index.save_local("faiss_index")
