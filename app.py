import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pipeline.rag_pipeline import create_embeddings, create_vector_store, create_rag_pipeline

# Load environment variables
load_dotenv()

# Load PDFs from local paths
pdf_path_1 = "docs/Blueprint-for-an-AI-Bill-of-Rights.pdf"
pdf_path_2 = "docs/NIST_AI_600-1.pdf"
loader1 = PyMuPDFLoader(pdf_path_1)
loader2 = PyMuPDFLoader(pdf_path_2)
documents1 = loader1.load()
documents2 = loader2.load()
documents = documents1 + documents2

# Split documents
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
documents = text_splitter.split_documents(documents)

# Set up embeddings (using ADA model for now)
EMBEDDING_MODEL = "text-embedding-ada-002"
embeddings = create_embeddings(model_name=EMBEDDING_MODEL)

# Create Qdrant vector database
vectorstore = create_vector_store(documents, embeddings)

# Create RAG pipeline using the retriever from the vector store
retriever = vectorstore.as_retriever()
rag_pipeline = create_rag_pipeline(retriever)

# Chainlit integration
@cl.on_message
async def on_message(message: cl.Message):
    # Prepare the input for the chain
    inputs = {"query": message.content}

    # Use the retrieval augmented QA chain
    response = await rag_pipeline.ainvoke(inputs)

    # Extract the content from the response
    result = response.content if hasattr(response, 'content') else str(response)

    # Send the result back to Chainlit
    await cl.Message(content=result).send()
