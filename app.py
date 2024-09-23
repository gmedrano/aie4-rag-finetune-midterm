import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate

# LCEL components
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter

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
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

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

# Set up the LLM
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define a function to extract the question
def get_question(inputs):
    return inputs["query"]

# Create the chain using LCEL syntax
retrieval_augmented_qa_chain = (
    # Step 1: Retrieve context based on the question
    {"context": RunnableLambda(get_question) | retriever, "question": itemgetter("query")}
    # Step 2: Assign the context back into inputs
    | RunnablePassthrough()
    # Step 3: Generate the response using the prompt and LLM
    | prompt
    | primary_qa_llm
)

# Chainlit integration
@cl.on_message
async def on_message(message: cl.Message):
    # Prepare the input for the chain
    inputs = {"query": message.content}

    # Use the retrieval augmented QA chain
    response = await retrieval_augmented_qa_chain.ainvoke(inputs)

    # Extract the content from the response
    result = response.content if hasattr(response, 'content') else str(response)

    # Send the result back to Chainlit
    await cl.Message(content=result).send()
