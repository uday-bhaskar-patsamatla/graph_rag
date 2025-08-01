# data_ingestion.py
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.chroma_db import insert_chunks_to_chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.chunks import chunks as document_chunks

from langchain.docstore.document import Document

# Import the Neo4j workflow class from the other file
from utils.neo4j_db import Neo4jWorkflow

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Get the API keys and Neo4j details from environment variables.
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

def load_and_split_documents(raw_text: str) -> List[str]:
    """
    Takes a single string of raw text and splits it into manageable chunks.
    This simulates the "cleaned chunks" a pipeline would produce.
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    # Split the raw text into a list of documents (chunks)
    return text_splitter.split_text(raw_text)

def run_data_ingestion():
    """
    Orchestrates the entire data ingestion process:
    1. Splits raw text into chunks.
    2. Calls the Neo4j pipeline to create a knowledge graph.
    3. Creates a ChromaDB collection from the same chunks.
    """
    print("--- Starting Data Ingestion Pipeline ---")
    documents = document_chunks
    # The sentences are the chunks from the text splitter
    neo4j_workflow = Neo4jWorkflow(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)

    neo4j_workflow.create_knowledge_base(documents)
    
    print("Neo4j knowledge graph creation complete.")

    # 3. Ingest into ChromaDB for vector search
    print("\n--- Starting ChromaDB Ingestion ---")
    

    chunks = [Document(page_content=doc, metadata={"doc_id": f"doc-{i+1}"}) for i, doc in enumerate(documents)]

    # 1. Insert chunks into Chroma DB
    try:
        print("Inserting chunks into ChromaDB...")
        chroma_db_instance = insert_chunks_to_chroma(chunks)

        if chroma_db_instance:
            print(f"ChromaDB collection '{chroma_db_instance.collection_name}' created successfully.")
        else:
            print("Failed to create ChromaDB collection.")
        
        
    
        # print("ChromaDB collection 'knowledge_chunks' created and populated successfully.")
    except Exception as e:
        print(f"Error creating ChromaDB collection: {e}")

    print("--- Data Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    run_data_ingestion()
