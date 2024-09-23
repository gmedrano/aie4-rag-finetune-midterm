import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Updated imports
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA  # Ensure this import is present

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

# Set up embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # Uses langchain-openai

# Create Qdrant vector database
LOCATION = ":memory:"
COLLECTION_NAME = "AI_Bill_of_Rights_and_NIST"
VECTOR_SIZE = 1536
qdrant_client = QdrantClient(":memory:")
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=LOCATION,
    collection_name=COLLECTION_NAME,
    distance_func="Cosine"
)

# Create retriever
retriever = vectorstore.as_retriever()

# Prompt template
template = """
Use the following context to answer the question. If the answer is not in the context, say 'I don't know.'

Question:
{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up the QA chain
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Uses langchain-openai
retrieval_augmented_qa_chain = RetrievalQA.from_chain_type(
    llm=primary_qa_llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Chainlit integration
@cl.on_message
async def on_message(message: cl.Message):
    # Use the retrieval augmented QA chain
    response = await retrieval_augmented_qa_chain.ainvoke({"query": message.content})

    # Check if response is a dict and extract 'result'
    if isinstance(response, dict) and 'result' in response:
        result = response['result']
    else:
        result = response  # In case the response is already a string

    # Send the result back to Chainlit
    await cl.Message(content=result).send()
