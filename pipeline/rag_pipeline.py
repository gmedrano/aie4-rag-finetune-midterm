from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter

# Set up embeddings (ADA or other fine-tuned model)
def create_embeddings(model_name="text-embedding-ada-002"):
    return OpenAIEmbeddings(model=model_name)

# Set up Qdrant vector database
def create_vector_store(documents, embeddings, collection_name="AI_Bill_of_Rights_and_NIST", location=":memory:"):
    qdrant_client = QdrantClient(location)
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        location=location,
        collection_name=collection_name,
        distance_func="Cosine"
    )
    return vectorstore

# Create RAG pipeline
def create_rag_pipeline(retriever, model_name="gpt-3.5-turbo"):
    # Define prompt template
    template = """
    Use the following context to answer the question. If the answer is not in the context, say 'I don't know.'

    Question:
    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Set up the language model
    primary_qa_llm = ChatOpenAI(model_name=model_name, temperature=0)

    # Define a function to extract the question
    def get_question(inputs):
        return inputs["query"]

    # Create the retrieval augmented QA chain
    retrieval_augmented_qa_chain = (
        {"context": RunnableLambda(get_question) | retriever, "question": itemgetter("query")}
        | RunnablePassthrough()
        | prompt
        | primary_qa_llm
    )

    return retrieval_augmented_qa_chain
