import os
import json
from typing import List, Dict, Any

from langchain_core.documents import Document as LC_Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
# from chromadb import Client, Settings
from pydantic import BaseModel, Field 
import chromadb
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CHROMADB_CLOUD_API_KEY = os.getenv("CHROMADB_CLOUD_API_KEY")
CHROMADB_CLOUD_TENANT = os.getenv("CHROMADB_CLOUD_TENANT")
CHROMADB_CLOUD_DATABASE = os.getenv("CHROMADB_CLOUD_DATABASE")


# --- Initialize Core Components ---

# OpenAI LLM for NER and Relationship Extraction
# Using gpt-4o-mini for cost-efficiency during ingestion, gpt-4o for higher accuracy if needed.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# OpenAI Embeddings for Vector Store
# text-embedding-3-small is recommended for its balance of performance and cost.
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Neo4j Graph Database Connection
# Ensure your Neo4j instance is running and accessible at NEO4J_URI
neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# ChromaDB Client (for persistent client-server mode)
# This is the recommended way to connect to a remote ChromaDB server for production.
# Ensure your ChromaDB server is running and accessible at CHROMADB_HOST:CHROMADB_PORT
try:
    # Using chromadb.CloudClient for cloud connection
    import chromadb.cloud as cloud
    chroma_client = cloud.CloudClient(
        api_key=CHROMADB_CLOUD_API_KEY,
        tenant=CHROMADB_CLOUD_TENANT,
        database=CHROMADB_CLOUD_DATABASE
    )
    # Get or create the collection for document chunks
    chroma_collection = chroma_client.get_or_create_collection(name="document_chunks")
    print(f"Successfully connected to ChromaDB Cloud (Tenant: {CHROMADB_CLOUD_TENANT}, Database: {CHROMADB_CLOUD_DATABASE})")
except Exception as e:
    print(f"ERROR: Could not connect to ChromaDB Cloud. Please ensure your API key, tenant, and database are correct. Error: {e}")
    # In a production system, you might want to exit or log this more severely.
    chroma_client = None # Set to None to prevent further errors
    chroma_collection = None


# LangChain's RecursiveCharacterTextSplitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Optimal chunk size often depends on data and LLM context window
    chunk_overlap=100, # Overlap to maintain context across chunk boundaries
    separators=["\n\n", "\n", " ", ""], # Prioritize splitting by paragraphs, then lines, then spaces
    length_function=len # Use character length for splitting
)

# --- Define Pydantic Models for Structured Extraction Output ---
# This schema guides the LLM to produce structured JSON for entities and relationships.
# Customize these models based on the specific entities and relationships in your complex data.
class ExtractedEntity(BaseModel):
    label: str = Field(description="Type of the entity, e.g., Person, Organization, Product, Concept.")
    name: str = Field(description="The exact text of the entity as it appears in the document.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties for the entity, e.g., {'title': 'CEO', 'industry': 'Tech'}")

class ExtractedRelationship(BaseModel):
    source_name: str = Field(description="Name of the source entity.")
    source_label: str = Field(description="Label of the source entity (must match an ExtractedEntity label).")
    target_name: str = Field(description="Name of the target entity.")
    target_label: str = Field(description="Label of the target entity (must match an ExtractedEntity label).")
    type: str = Field(description="Type of the relationship, e.g., WORKS_FOR, LOCATED_IN, PART_OF, DEVELOPS.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the relationship, e.g., {'start_date': '2022-01-01'}")

class DocumentExtraction(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list, description="List of extracted entities.")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="List of extracted relationships.")

# Convert Pydantic model to JSON Schema for LLM instruction
# This is how we tell the LLM the exact structure we expect.
# NER_SCHEMA_JSON = json.dumps(DocumentExtraction.model_json_schema(), indent=2) # No longer directly used in prompt with with_structured_output

# --- Prompt Template for NER and Relationship Extraction ---
# This prompt instructs the LLM on how to perform the extraction.
# Few-shot examples (not included here for brevity but highly recommended)
# would go in the prompt to demonstrate the desired output format and quality.
# The prompt is simplified as with_structured_output handles schema adherence.
ERE_PROMPT_TEMPLATE = """
You are an expert information extraction system. Your task is to extract entities and their relationships
from the provided text.

Only extract entities and relationships that are explicitly mentioned or strongly implied in the text.
Do not hallucinate any information.

Text to process:
---
{text_chunk}
---
"""

# --- Document Processing Function ---
def process_document_pipeline(document_id: str, document_content: str, document_metadata: Dict[str, Any]):
    """
    Orchestrates the entire ingestion pipeline for a single document:
    1. Stores document metadata in Neo4j.
    2. Chunks the document.
    3. For each chunk:
        a. Performs NER and relationship extraction using LLM.
        b. Populates Neo4j with chunk nodes, extracted entities, and relationships.
        c. Generates embeddings.
        d. Stores chunk and embedding in ChromaDB.
    """
    print(f"\n--- Processing document: {document_id} ---")

    # 1. Store Document Node in Neo4j (Idempotent MERGE)
    try:
        neo4j_graph.query(
            "MERGE (d:Document {id: $document_id}) "
            "SET d += $metadata", # Set all metadata as properties
            params={"document_id": document_id, "metadata": document_metadata}
        )
        print(f"Neo4j: Document '{document_id}' node MERGED.")
    except Exception as e:
        print(f"Error merging Document node in Neo4j for {document_id}: {e}")
        return # Stop processing this document if initial merge fails

    # 2. Chunk the document
    chunks = text_splitter.split_text(document_content)
    print(f"Document chunked into {len(chunks)} parts.")

    # Prepare lists for batching ChromaDB additions
    chroma_chunk_ids = []
    chroma_chunk_texts = []
    chroma_chunk_metadatas = []

    previous_chunk_id = None

    # Create a structured LLM instance for reliable JSON output
    # This is the key change to ensure valid Pydantic model output
    structured_llm = llm.with_structured_output(schema=DocumentExtraction)

    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{i}"
        print(f"  Processing chunk: {chunk_id}")

        # 3a. NER and Relationship Extraction for the current chunk
        extracted_entities: List[ExtractedEntity] = []
        extracted_relationships: List[ExtractedRelationship] = []
        try:
            # Invoke the structured LLM directly with the chunk text
            # The structured_llm will handle the schema and ensure valid output
            parsed_extraction: DocumentExtraction = structured_llm.invoke({"text_chunk": chunk_text})
            
            extracted_entities = parsed_extraction.entities
            extracted_relationships = parsed_extraction.relationships

            print(f"    Extracted {len(extracted_entities)} entities and {len(extracted_relationships)} relationships.")

        except Exception as e:
            print(f"    Error during NER extraction for chunk {chunk_id}: {e}")
            # In a real production system, you might want to log the raw LLM response here
            # to debug why it failed to conform to the schema.
            # print(f"    LLM Raw Output (if available): {llm_response.content[:500]}...")

        # 3b. Populate Neo4j with Chunk, Entities, and Relationships
        try:
            # Create/Merge Chunk Node
            neo4j_graph.query(
                "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                "SET c.text = $chunk_text, c.document_id = $document_id, c.chunk_index = $chunk_index",
                params={"chunk_id": chunk_id, "chunk_text": chunk_text, "document_id": document_id, "chunk_index": i}
            )
            # Link Chunk to Parent Document
            neo4j_graph.query(
                "MATCH (d:Document {id: $document_id}), (c:Chunk {chunk_id: $chunk_id}) "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                params={"document_id": document_id, "chunk_id": chunk_id}
            )
            # Link Sequential Chunks (for document flow/context)
            if previous_chunk_id:
                neo4j_graph.query(
                    "MATCH (prev_c:Chunk {chunk_id: $previous_chunk_id}), (current_c:Chunk {chunk_id: $current_chunk_id}) "
                    "MERGE (prev_c)-[:NEXT_CHUNK]->(current_c)",
                    params={"previous_chunk_id": previous_chunk_id, "current_chunk_id": chunk_id}
                )
            previous_chunk_id = chunk_id

            # Create/Merge Entities and Link to Chunk
            for entity in extracted_entities:
                # Sanitize label for Neo4j (e.g., remove spaces or special chars if present)
                node_label = entity.label.replace(" ", "_").replace("-", "_")
                neo4j_graph.query(
                    f"MERGE (e:{node_label} {{name: $entity_name}}) "
                    f"SET e += $properties "
                    f"WITH e MATCH (c:Chunk {{chunk_id: $chunk_id}}) "
                    f"MERGE (c)-[:MENTIONS]->(e)", # Link chunk to entity
                    params={"entity_name": entity.name, "properties": entity.properties, "chunk_id": chunk_id}
                )

            # Create/Merge Relationships
            for rel in extracted_relationships:
                # Sanitize labels and type for Neo4j
                source_label = rel.source_label.replace(" ", "_").replace("-", "_")
                target_label = rel.target_label.replace(" ", "_").replace("-", "_")
                rel_type = rel.type.replace(" ", "_").replace("-", "_")

                neo4j_graph.query(
                    f"MATCH (s:{source_label} {{name: $source_name}}), "
                    f"(t:{target_label} {{name: $target_name}}) "
                    f"MERGE (s)-[r:{rel_type}]->(t) "
                    f"SET r += $properties",
                    params={
                        "source_name": rel.source_name,
                        "target_name": rel.target_name,
                        "properties": rel.properties
                    }
                )
        except Exception as e:
            print(f"    Error populating Neo4j for chunk {chunk_id}: {e}")

        # 3c. Prepare chunk data for Embedding and ChromaDB storage
        chroma_chunk_ids.append(chunk_id)
        chroma_chunk_texts.append(chunk_text)
        chroma_chunk_metadatas.append({
            "document_id": document_id,
            "chunk_index": i,
            "title": document_metadata.get("title"), # Include original doc title
            "category": document_metadata.get("category"), # Include original doc category
            # FIX: Join entities_mentioned list into a string for ChromaDB metadata compatibility
            "entities_mentioned": ", ".join([e.name for e in extracted_entities]) 
        })
    
    # 3d. Generate Embeddings and Store in ChromaDB (Batching for efficiency)
    if chroma_chunk_texts and chroma_collection: # Only proceed if ChromaDB connection is successful
        print(f"  Generating embeddings for {len(chroma_chunk_texts)} chunks and storing in ChromaDB...")
        try:
            chunk_embeddings = embeddings_model.embed_documents(chroma_chunk_texts)
            chroma_collection.add(
                documents=chroma_chunk_texts,
                embeddings=chunk_embeddings,
                metadatas=chroma_chunk_metadatas,
                ids=chroma_chunk_ids
            )
            print(f"  ChromaDB: {len(chroma_chunk_ids)} chunks added for document '{document_id}'.")
        except Exception as e:
            print(f"  Error adding chunks to ChromaDB for document {document_id}: {e}")
    elif not chroma_collection:
        print(f"  Skipping ChromaDB operations for document '{document_id}' due to connection error.")
    else:
        print(f"  No chunks to add to ChromaDB for document '{document_id}'.")

    print(f"--- Document {document_id} processing complete. ---")


# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    # --- IMPORTANT: Run these Cypher commands in Neo4j Browser/Client ONCE to create indexes ---
    # These indexes are crucial for performance, especially for MERGE operations.
    # Replace labels (e.g., Person, Organization) with the actual entity labels you define in your schema.
    print("Ensuring Neo4j indexes and constraints are in place...")
    try:
        # Corrected Cypher syntax for creating constraints
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE")
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE")
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (prod:Product) REQUIRE prod.name IS UNIQUE")
        neo4j_graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (con:Concept) REQUIRE con.name IS UNIQUE")
        # Add more constraints for other entity types you define in ExtractedEntity
        print("Neo4j indexes and constraints checked/created successfully.")
    except Exception as e:
        print(f"Error creating Neo4j indexes/constraints. Ensure Neo4j is running and accessible: {e}")
        # Exit or handle appropriately if DB setup fails
        exit()

    # --- Sample Document (Replace with your actual cleaned documents) ---
    sample_document_1_content = """
    Dr. Anya Sharma, a leading AI researcher, joined Quantum Innovations in 2022.
    She previously worked at Alpha Research, located in Bangalore, India, from 2018 to 2021.
    Quantum Innovations, headquartered in San Francisco, develops cutting-edge machine learning algorithms.
    Their flagship product, NeuroLink, was launched in June 2023.
    This document also discusses the applications of NeuroLink in healthcare.
    John Doe, CEO of Alpha Research, announced a new partnership with Global Tech, based in New York.
    """
    sample_document_1_metadata = {"title": "AI Research Overview", "category": "Technology", "source_file": "doc1.txt"}

    sample_document_2_content = """
    The recent economic downturn has impacted several industries, including manufacturing and retail.
    Analysts at Market Insights predict a recovery starting in Q3 2024.
    Companies like MegaCorp, a multinational conglomerate, are diversifying their portfolios.
    MegaCorp acquired Innovate Solutions, a software firm, in January 2024.
    The acquisition was finalized in London, UK. This marks a significant shift in their strategy.
    """
    sample_document_2_metadata = {"title": "Economic Outlook 2024", "category": "Finance", "source_file": "doc2.txt"}

    # --- Process Sample Documents ---
    process_document_pipeline("doc_001", sample_document_1_content, sample_document_1_metadata)
    process_document_pipeline("doc_002", sample_document_2_content, sample_document_2_metadata)

    # --- Example of retrieving from ChromaDB (after ingestion) ---
    print("\n--- Example ChromaDB Query ---")
    query_text = "Who works at Alpha Research?"
    if chroma_collection: # Only attempt query if collection was initialized
        try:
            # Generate embedding for the query
            query_embedding = embeddings_model.embed_query(query_text)
            
            # Query ChromaDB for similar chunks
            results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5, # Number of top similar chunks to retrieve
                include=['documents', 'metadatas', 'distances']
            )
            print(f"ChromaDB results for '{query_text}':")
            for i in range(len(results['ids'][0])):
                print(f"  Chunk ID: {results['ids'][0][i]}, Distance: {results['distances'][0][i]:.4f}")
                print(f"    Text: {results['documents'][0][i][:100]}...") # Print first 100 chars
                print(f"    Metadata: {results['metadatas'][0][i]}")
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
    else:
        print("ChromaDB not connected, skipping query example.")

    # --- Example of querying Neo4j (after ingestion) ---
    print("\n--- Example Neo4j Query ---")
    try:
        cypher_query = """
        MATCH (p:Person)-[:WORKS_FOR]->(o:Organization)
        WHERE o.name = 'Alpha Research'
        RETURN p.name AS PersonName, o.name AS OrganizationName
        """
        neo4j_results = neo4j_graph.query(cypher_query)
        print(f"Neo4j results for 'Persons working at Alpha Research': {neo4j_results}")
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
