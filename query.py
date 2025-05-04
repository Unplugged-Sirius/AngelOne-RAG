from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# Check if FAISS index exists
index_name = "faiss_index"
if not os.path.exists(index_name):
    print(f"Error: FAISS index '{index_name}' not found. Please create it first.")
    print("Example: faiss_index = FAISS.from_texts(texts, embedding_model)")
    print("         faiss_index.save_local('faiss_index')")
    exit(1)

# Define prompt template
prompt_template = """
You are a helpful customer support assistant for Angel One.

Here is the context:
{context}
Answer the question below using only the information provided in the context.
If you don't see relevent content in the context, say "I don't know."

Question: {question}
Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Load embedding model
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
try:
    print(f"Loading FAISS index from '{index_name}'...")
    faiss_index = FAISS.load_local(
        index_name,
        embedding_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit(1)

# Load LLM pipeline
print("Loading language model...")
try:
    llm_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
except Exception as e:
    print(f"Error loading language model: {e}")
    exit(1)

# Create retriever and RAG pipeline
print("Setting up RAG pipeline...")
retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
qa_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False
)

def answer_query(query: str):
    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        result = qa_chain.run({"context": context, "question": query}).strip()
        
        if not result:
            return "I don't know enough to answer that question based on my knowledge."
        
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

print("RAG assistant ready! Type 'exit' or 'quit' to end.")
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break
    response = answer_query(question)
    print("Bot:", response)