import gradio as gr
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings 
# Removed: from langchain.llms import HuggingFacePipeline 
# Removed: from langchain.chains import ConversationChain # Not used in RAG
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub # We will rely only on this

from pathlib import Path
import chromadb


# --- OPTIMIZED LIST OF LLMS FOR FREE HF SPACE DEPLOYMENT ---
# 1. Models that have robust free Inference API endpoints (recommended for performance).
# 2. Very small models that might run on the CPU (e.g., TinyLlama) as a last resort.
list_llm = [
    # Free Inference API: Excellent, stable performance.
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-7b-chat-hf",
    "google/gemma-7b-it",
    
    # Very Small Local Model (Fallback/Slow CPU run)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load PDF
def load_pdf(file_path):
    """Loads a PDF file using PyPDFLoader."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

# Split documents into chunks
def split_documents(documents, chunk_size, chunk_overlap):
    """Splits documents into smaller chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    return texts

# Initialize the vector database (Chroma)
def initialize_database(document_path, chunk_size, chunk_overlap):
    """Initializes the Chroma vector store with document embeddings."""

    # 1. Check for document
    if not document_path:
        # Return 3 values matching the outputs
        return None, "", "Database initialization failed: No document provided."

    # 2. Generate collection name
    collection_name = "rag_collection_" + os.path.basename(document_path).split('.')[0]

    # 3. Load and split
    try:
        documents = load_pdf(document_path)
    except Exception as e:
        print(f"Error in load_pdf: {e}")
        return None, collection_name, "Database initialization failed: Error loading PDF content."

    if not documents:
        return None, collection_name, "Database initialization failed: Could not load document content."

    texts = split_documents(documents, chunk_size, chunk_overlap)

    # 4. Initialize Embeddings
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # 5. Create Vector Store (Chroma)
    client = chromadb.Client()

    # Yield a status update (3 values)
    yield None, collection_name, "Embedding... (This may take a moment)"

    try:
        # Create and persist the Chroma database
        vector_db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )
        # Return 3 values
        return vector_db, collection_name, f"Database initialized with {len(texts)} chunks."
    except Exception as e:
        print(f"Error during vector store creation: {e}")
        # Return 3 values
        return None, collection_name, "Database initialization failed: Error during embedding process."


# Initialize the LLM and the QA chain
def initialize_LLM(llm_name, temperature, max_new_tokens, top_k, vector_db):
    """Initializes the HuggingFace LLM using the external HuggingFaceHub API."""

    # 1. Check for DB
    if not vector_db:
        # Return 2 values matching the outputs
        return None, "Error: Vector database not initialized. Please upload and process a document first."

    # Yield a status update (2 values)
    yield None, "Connecting to HuggingFace Inference API..."

    # The model ID is the selected model name
    model_id = list_llm[list_llm_simple.index(llm_name)]

    try:
        llm = HuggingFaceHub(
            repo_id=model_id,
            # Note: This requires the HUGGINGFACE_TOKEN secret to be set in your Space.
            model_kwargs={
                "temperature": temperature, 
                "max_length": max_new_tokens
            }
        )
    except Exception as e:
        print(f"HuggingFaceHub error for {model_id}: {e}")
        # Return 2 values
        return None, f"Could not initialize LLM {model_id}. Check your HuggingFace token and model accessibility."

    # Yield a status update (2 values)
    yield None, "Initializing Chain..."

    # Initialize Conversational Retrieval Chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        return_source_documents=True,
        chain_type="stuff", # 'stuff' works well for single-document RAG
        verbose=True
    )

    # Return 2 values
    return qa_chain, f"Chain initialized with {model_id}."

# Main conversation function
def conversation(qa_chain, message, history):
    """Handles the user message and generates a response from the QA chain."""
    if not qa_chain:
        # Return error state if the chain is not ready
        return None, "", history + [[message, "Error: The LLM/QA Chain is not initialized. Please click 'Initialize LLM & Chain' first."]], "", 0, "", 0
        
    # Get response from the QA chain
    result = qa_chain({"question": message})
    
    # Extract response and source documents
    response = result["answer"]
    source_documents = result.get("source_documents", [])
    
    # Format source documents for display
    doc_source1 = ""
    source1_page = 0
    doc_source2 = ""
    source2_page = 0
    
    if source_documents:
        # Source 1
        doc1 = source_documents[0]
        doc_source1 = doc1.metadata.get('source', 'N/A').split('/')[-1]
        source1_page = doc1.metadata.get('page', 'N/A')
        
        # Source 2 (if available)
        if len(source_documents) > 1:
            doc2 = source_documents[1]
            doc_source2 = doc2.metadata.get('source', 'N/A').split('/')[-1]
            source2_page = doc2.metadata.get('page', 'N/A')

    # Append to history
    history.append([message, response])
    
    # Return updated chain, clear message, updated history, and sources
    return qa_chain, "", history, doc_source1, source1_page, doc_source2, source2_page


# Gradio Interface
with gr.Blocks(title="Conversational RAG Application, using PDF", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Conversational RAG with Open-Source LLMs (Optimized for Free HF Space)")
    gr.Markdown("Upload a PDF, initialize the database and LLM, and start asking questions. **Note:** This version relies on the HuggingFace Inference API, which requires setting your `HUGGINGFACE_TOKEN` in the Space secrets.")
    
    # Hidden states to maintain the LLM chain and vector database
    vector_db = gr.State(None)
    qa_chain = gr.State(None)
    collection_name = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1. Document & ChromaDB Setup") 
            document = gr.File(label="Upload PDF Document - Drop or Click", file_types=[".pdf"], type="filepath")
            
            with gr.Accordion("Chunking Parameters", open=False):
                slider_chunk_size = gr.Slider(minimum=100, maximum=2000, value=1000, step=100, label="Chunk Size")
                slider_chunk_overlap = gr.Slider(minimum=0, maximum=500, value=200, step=50, label="Chunk Overlap")
            
            db_btn = gr.Button("Initialize ChromaDB (Embed & Store)", variant="primary") 
            db_progress = gr.Text(label="DB Status", value="Waiting for file upload...")

        with gr.Column(scale=1):
            gr.Markdown("### Step 2. LLM & Chain Setup")
            llm_btn = gr.Dropdown(list_llm_simple, label="Select LLM (Uses external API)", value=list_llm_simple[0])
            
            with gr.Accordion("LLM Parameters", open=False):
                slider_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="Temperature (Creativity)")
                slider_maxtokens = gr.Slider(minimum=128, maximum=2048, value=512, step=128, label="Max New Tokens")
                slider_topk = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K (for sampling)")
            
            qachain_btn = gr.Button("Initialize LLM & Chain", variant="primary")
            llm_progress = gr.Text(label="LLM Status", value="Awaiting database initialization...")
            
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Step 3. Conversation")
            # --- START: CHAT HISTORY  ---
            chatbot = gr.Chatbot(label="Chat History", height=450)
            # --- END: CHAT HISTORY ---
            
            # --- START: USER INPUT  ---
            msg = gr.Textbox(label="Your Question", placeholder="Ask a question about the document...")
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot])
            # --- END: USER INPUT ---
            
        with gr.Column(scale=1):
            gr.Markdown("### Step 4. Sources")
            doc_source1 = gr.Textbox(label="Source 1 Document Name", interactive=False)
            source1_page = gr.Number(label="Source 1 Page", precision=0, interactive=False)
            doc_source2 = gr.Textbox(label="Source 2 Document Name", interactive=False)
            source2_page = gr.Number(label="Source 2 Page", precision=0, interactive=False)

        # Preprocessing events
        db_btn.click(initialize_database, 
            inputs=[document, slider_chunk_size, slider_chunk_overlap], 
            outputs=[vector_db, collection_name, db_progress])
            
        qachain_btn.click(initialize_LLM, 
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], 
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0], 
            inputs=None, 
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page], 
            queue=False)

        # Chatbot events
        msg.submit(conversation, 
            inputs=[qa_chain, msg, chatbot], 
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page], 
            queue=False)
            
        submit_btn.click(conversation, 
            inputs=[qa_chain, msg, chatbot], 
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page], 
            queue=False)
            
        clear_btn.click(lambda:[None,"",0,"",0], 
            inputs=None, 
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page], 
            queue=False)

demo.launch(share=True)