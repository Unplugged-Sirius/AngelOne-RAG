import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.llms import HuggingFacePipeline

# Check for FAISS index
index_name = "faiss_index"
if not os.path.exists(index_name):
    st.error(f"FAISS index '{index_name}' not found. Please create it first.")
    st.stop()

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
@st.cache_resource
def load_faiss_index(_embedding_model):
    return FAISS.load_local(
        index_name,
        _embedding_model,
        allow_dangerous_deserialization=True
    )

# Load LLM with memory optimization
@st.cache_resource
def load_llm():
    # Check if CUDA is available and set device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        st.info("Using GPU for inference")
    else:
        st.info("Using CPU for inference")
    
    # Use a more memory-efficient approach
    try:
        # Option 1: Try with a smaller Mistral model with bitsandbytes quantization
        if device == "cuda":
            model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"  # GPTQ quantized model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                revision="main"
            )
        else:
            # For CPU, use a smaller model
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        
        return HuggingFacePipeline(pipeline=pipe)
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        
        # Fallback to an even smaller model
        st.warning("Falling back to a smaller model due to memory constraints.")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
        
        return HuggingFacePipeline(pipeline=pipe)

# Prompt
prompt_template = """
You are a helpful customer support assistant for Angel One.

Here is the context:
{context}
Answer the question below using only the information provided in the context.
If you don't see relevant content in the context, say "I don't know."

Question: {question}

Provide a concise and direct answer without mentioning that you're looking at context or repeating my question.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Store initialization status
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# UI
st.title("📊 Angel One RAG Assistant")
st.markdown("Ask your question below:")

# Initialize components
if not st.session_state.initialized:
    with st.spinner("Loading models... This may take a few minutes on first run"):
        try:
            # Load models and index
            embedding_model = load_embedding_model()
            faiss_index = load_faiss_index(embedding_model)
            llm = load_llm()
            retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
            qa_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
            
            # Store in session state
            st.session_state.embedding_model = embedding_model
            st.session_state.faiss_index = faiss_index
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.qa_chain = qa_chain
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

# Query function
def answer_query(query: str):
    try:
        docs = st.session_state.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a formatted prompt but don't include it in the response
        formatted_prompt = f"""
You are a helpful customer support assistant for Angel One.

Here is the context:
{context}

Answer the question below using only the information provided in the context.
If you don't see relevant content in the context, say "I don't know."

Question: {query}

Provide a concise answer without repeating the question or mentioning the context.
"""
        # Use a direct call to the LLM to avoid exposing the template
        # This bypasses the LLMChain and uses a direct call to the pipeline
        inputs = st.session_state.llm.pipeline.tokenizer(
            formatted_prompt, 
            return_tensors="pt"
        )
        inputs = {k: v.to(st.session_state.llm.pipeline.model.device) for k, v in inputs.items()}
        
        # Generate response
        output = st.session_state.llm.pipeline.model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # Decode the response and extract just the answer part
        full_response = st.session_state.llm.pipeline.tokenizer.decode(
            output[0], skip_special_tokens=True
        )
        
        # Extract just the answer part (after the prompt)
        answer = full_response.split("Provide a concise answer without repeating the question or mentioning the context.")[-1].strip()
        
        # If the answer appears to be the model just repeating instructions, use a simpler approach
        if "I don't know" not in answer and len(answer) < 10:
            # Fall back to the original method but try to clean up the result
            result = st.session_state.qa_chain.run({"context": context, "question": query}).strip()
            # Try to clean up common prefixes
            for prefix in ["Answer:", "The answer is:", "Based on the context,"]:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()
            return result
            
        return answer if answer else "I don't know."
    except Exception as e:
        st.error(f"Error: {str(e)}")
        # Fallback to original method if direct generation fails
        try:
            result = st.session_state.qa_chain.run({"context": context, "question": query}).strip()
            return result
        except:
            return "An error occurred while processing your question. Please try again."






# User input
user_input = st.text_input("You:", placeholder="Type your question here...")
submit_button = st.button("Submit")

if submit_button and user_input:
    with st.spinner("Thinking..."):
        response = answer_query(user_input)
        
        # Clean up the response to remove any prompt artifacts
        response_lines = response.split('\n')
        
        # Remove any lines that might be part of the prompt
        clean_lines = []
        for line in response_lines:
            # Skip lines that seem to be part of the prompt or instruction
            if any(x in line.lower() for x in ["context:", "question:", "answer:", "you are", "helpful assistant"]):
                continue
            clean_lines.append(line)
        
        clean_response = '\n'.join(clean_lines).strip()
        
        # If nothing left after cleaning, use original response
        if not clean_response:
            clean_response = response
    
    st.markdown("### 🤖 Bot:")
    st.write(clean_response)