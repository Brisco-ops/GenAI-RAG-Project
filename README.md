---
title: My PDF Chatbot
emoji:  🤖
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: true 
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference 

---
1. The concept behind the application
Our project is a conversational RAG (Retrieval-Augmented Generation) application. Its purpose is to enable users to “chat” with a PDF document that they upload.

The concept works in three stages:

Ingestion (Pre-processing): The user uploads a PDF file. The application cuts it into small pieces of text (chunks) that overlap.

Embedding (Learning): Each piece of text is transformed into a mathematical representation (a “vector”) using an embedding model (such as all-MiniLM-L6-v2). These vectors are stored in a vector database (ChromaDB).

Generation (Conversation):

When a user asks a question, the application converts that question into a vector.

It compares this vector to those in the database to find the most relevant chunks of text from the PDF.

It sends these relevant pieces (the “context”) to a large external language model (LLM) (such as Mistral or Gemma via the Hugging Face API), asking it to answer the question based solely on this context.

The interface, built with Gradio, allows this flow to be managed interactively.



# Local installation and execution

Follow these steps to launch the project on your machine.

### Prerequisites

* Python 3.9+
* Git

### 1. Clone the repository

```bash
git clone [https://github.com/Brisco-ops/GenAI-RAG-Project.git](https://github.com/Brisco-ops/GenAI-RAG-Project.git)
cd GenAI-RAG-Project

2. Create a virtual environment

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

3. Intall dependencies
gradio
langchain==0.0.354
pypdf
chromadb
sentence-transformers
huggingface-hub

# Note: The langchain version is fixed at 0.0.354 because newer versions have major changes (imports) that would break the code.

4. Set up your “Secret” (Hugging Face Token)

The application uses Hugging Face's external API (HuggingFaceHub) to connect to LLMs. This requires an Access Token.

- Generate a token on your Hugging Face settings page (give it the read or write role).

- You must set this token as an environment variable before launching the application.

# On macOS / Linux
export HUGGINGFACE_TOKEN='hf.....'

# On Windows (PowerShell)
$env:HUGGINGFACE_TOKEN='hf....'


5 Launch the Application 

gradio app.py

Open the URL displayed in the terminal in your browser.




# 3. Technical choices and limitations

## Technical choices

User interface (UI): Gradio

Why?It is the standard for quickly creating AI demos. It is native to the Hugging Face (HF) ecosystem and simplifies the creation of interactive interfaces (sliders, file uploads, chatbots).

Orchestration (RAG Logic): LangChain

Why? It is the dominant framework for assembling the components of a RAG application. 

We use it for 'PyPDFLoader' (loading PDFs), 'RecursiveCharacterTextSplitter' (managing chunks and overlaps), and 'ConversationalRetrievalChain' (managing memory and RAG).

LLM (The “brain”): 'HuggingFaceHub' (External API)

Why?This was the most critical choice. Instead of trying to load a 7 billion parameter LLM (e.g., Mistral-7B) on a CPU (which is impossible), we use the HF API. This offloads the heavy calculations to HF's GPUs, leaving only light tasks for our “Space.”

Embedding (The “understanding”): sentence-transformers/all-MiniLM-L6-v2

Why? It's a very powerful embedding model, but small enough to run on a CPU in a (relatively) reasonable amount of time.

Vector database: 'ChromaDB'

Why? It is a lightweight, popular vector database that integrates seamlessly with LangChain without requiring a separate database server.



## Limitations

Slow Embedding (Bottleneck):

The “Initialize ChromaDB” step is the biggest limitation. Embedding calculations (even with 'all-MiniLM') are very slow on HF's free “Basic CPU” hardware. A 10 MB PDF can take more than 10-15 minutes, which makes it seem like the application is “frozen” while it is working.

Stateless Application:

The application is “stateless.” Each time it is restarted or a new session is opened, the user must re-download and re-process the PDF. The embedding is not saved. For a production application, we would “pre-compute” the embeddings and store them persistently.

Token dependency:

The application will not work at all if the 'HUGGINGFACE_TOKEN' secret is not correctly configured in the “Space,” as it cannot call the LLM API.

LangChain Version Dependency:

We had to force the installation of an older version of LangChain ('0.0.354') because the API changed drastically, and our code is based on the old “imports” ('langchain.document_loaders').
