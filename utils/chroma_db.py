# chroma_operations.py
import os
from typing import List, Optional
from dotenv import load_dotenv

import chromadb
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Get the API key and Chroma DB details from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Use the specific Chroma Cloud configuration provided by the user
CHROMA_CLOUD_HOST = os.getenv("CHROMA_API_HOST", "api.trychroma.com")
CHROMA_CLOUD_TENANT = os.getenv("CHROMADB_CLOUD_TENANT")
CHROMA_CLOUD_DATABASE = os.getenv("CHROMADB_CLOUD_DATABASE")
CHROMA_API_TOKEN = os.getenv("CHROMADB_CLOUD_API_KEY")

if not all([CHROMA_API_TOKEN]):
    print("Warning: Chroma Cloud credentials are not fully configured. Falling back to a local client for demonstration.")
    USE_CLOUD = False
else:
    USE_CLOUD = True
    print("Using Chroma Cloud client.")

COLLECTION_NAME = "Graph_RAG_Apple_corpus"
# Instantiate the embedding model. This will now use OpenAI's API.
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_chroma_client():
    """Returns a ChromaDB client, either cloud or local."""
    if USE_CLOUD:
        # Use the specific connection details for the cloud client
        return chromadb.HttpClient(
            ssl=True,
            host=CHROMA_CLOUD_HOST,
            tenant=CHROMA_CLOUD_TENANT,
            database=CHROMA_CLOUD_DATABASE,
            headers={'x-chroma-token': CHROMA_API_TOKEN}
        )
    else:
        # Fallback to a local client if cloud details are not provided
        return chromadb.PersistentClient(path="./local_chroma_db")


def insert_chunks_to_chroma(
    chunks: List[Document],
    collection_name: str = COLLECTION_NAME
) -> Chroma:
    """
    Inserts a list of document chunks into a Chroma DB vector store.
    This function creates the embeddings and the Chroma collection.

    Args:
        chunks (List[Document]): A list of Document objects, each with a text chunk
                                  and optional metadata (e.g., 'doc_id').
        collection_name (str): The name of the collection to use.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    print("Initializing Chroma DB and creating embeddings...")
    try:
        client = get_chroma_client()
        # The Chroma.from_documents method now uses the client to connect
        # to the Chroma cloud instance.
        chroma_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            client=client,
            collection_name=collection_name,
        )
        print(f"Successfully inserted {len(chunks)} chunks into Chroma DB.")
        return chroma_db
    except Exception as e:
        print(f"An error occurred while inserting chunks into Chroma DB: {e}")
        return None

def retrieve_from_chroma(
    query: str,
    collection_name: str = COLLECTION_NAME,
    k: int = 3
) -> Optional[List[Document]]:
    """
    Retrieves the most relevant chunks from Chroma DB based on a user query.

    Args:
        query (str): The user query string.
        collection_name (str): The name of the Chroma collection.
        k (int): The number of relevant chunks to retrieve.

    Returns:
        Optional[List[Document]]: A list of retrieved relevant Document objects,
                                  or None if the database cannot be loaded.
    """
    print(f"\nRetrieving relevant chunks for the query: '{query}'")
    try:
        client = get_chroma_client()
        
        # Load the existing collection from the Chroma cloud instance
        chroma_db = Chroma(
            client=client,
            embedding_function=embedding_function,
            collection_name=collection_name
        )

        # Perform a similarity search on the Chroma DB
        retrieved_docs = chroma_db.similarity_search(query, k=k)
        print(f"Retrieved {len(retrieved_docs)} chunks from Chroma DB.")
        return retrieved_docs
    except Exception as e:
        print(f"An error occurred during retrieval from Chroma DB: {e}")
        return None

# if __name__ == "__main__":
#     # --- Example Usage ---
    
#     # Simulate document chunks with metadata (e.g., a unique ID)
#     documents = chunks
#     chunks = [Document(page_content=doc, metadata={"doc_id": f"doc-{i+1}"}) for i, doc in enumerate(documents)]

#     # 1. Insert chunks into Chroma DB
#     chroma_db_instance = insert_chunks_to_chroma(chunks)

#     if chroma_db_instance:
#         # 2. Retrieve relevant chunks for a query
#         user_query = "Who is the founder of Apple?"
#         retrieved_docs = retrieve_from_chroma(user_query, k=2)

#         if retrieved_docs:
#             print("\n--- Retrieved Documents from Chroma ---")
#             for i, doc in enumerate(retrieved_docs):
#                 print(f"Rank {i+1}: ID={doc.metadata.get('doc_id')}\nContent: {doc.page_content}\n")
#     else:
#         print("Chroma DB insertion failed, cannot proceed with retrieval.")
