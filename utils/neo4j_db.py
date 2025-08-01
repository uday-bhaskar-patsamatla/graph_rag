# neo4j_workflow.py
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain_neo4j import Neo4jGraph
from chunks import chunks as document_chunks

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Get the API keys and Neo4j details from environment variables.
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY]):
    raise ValueError("Neo4j cloud connection details (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) not found in environment variables.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

class Neo4jWorkflow:
    """
    Manages the complete knowledge graph workflow, from ingestion to retrieval.
    This class handles Neo4j connections and interacts with the OpenAI API for NER.
    """
    def __init__(self, uri: str, user: str, password: str, openai_api_key: str):
        """
        Initializes the Neo4jGraph connection and OpenAI client.
        """
        # The Neo4jGraph class from LangChain manages the connection internally.
        self.neo4j_graph = Neo4jGraph(url=uri, username=user, password=password)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.uri = uri
        self.user = user
        print(f"Neo4jGraph connection initialized for cloud URI: {self.uri}")

    def _extract_entities_for_creation(self, text: str) -> List[Dict[str, Any]]:
        """
        Sub-function to extract entities and relationships from a text string for graph creation.
        This uses a structured JSON output format to ensure reliable parsing.
        """
        prompt = f"""
        You are an expert in Named Entity Recognition and relationship extraction.
        Your task is to identify entities and their relationships from the given text.
        
        Extract entities and relationships into a structured JSON format. 
        Each object in the JSON array should contain:
        - "entity1": The name of the source entity.
        - "entity1_label": The label/type of the source entity (e.g., "Person", "Company").
        - "relationship": The type of relationship connecting the two entities (e.g., "FOUNDED_BY", "WORKS_AT").
        - "entity2": The name of the target entity.
        - "entity2_label": The label/type of the target entity.
        
        Example JSON output:
        [
          {{
            "entity1": "Apple Inc.",
            "entity1_label": "Company",
            "relationship": "FOUNDED_BY",
            "entity2": "Steve Jobs",
            "entity2_label": "Person"
          }}
        ]
        
        If no clear relationships are found, return an empty JSON array.
        
        Here is the text to analyze:
        "{text}"
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            # The LLM's response might include Markdown. We need to extract the raw JSON.
            if response_content.startswith("```json"):
                response_content = response_content.strip("```json\n").strip("\n```")

            return json.loads(response_content)
        except Exception as e:
            print(f"Error during NER extraction for creation: {e}")
            return []

    def _extract_entities_for_query(self, text: str) -> List[Dict[str, Any]]:
        """
        Sub-function to extract only entities from a query string for retrieval.
        This uses a simpler JSON output format.
        """
        prompt = f"""
        You are an expert in Named Entity Recognition.
        Your task is to identify all key entities from the given query.
        
        Extract entities into a structured JSON format. 
        Each object in the JSON array should contain:
        - "name": The name of the entity (e.g., "Apple", "Steve Jobs").
        - "label": The label/type of the entity (e.g., "Company", "Person").
        
        Example JSON output:
        [
          {{
            "name": "Apple",
            "label": "Company"
          }},
          {{
            "name": "Steve Jobs",
            "label": "Person"
          }}
        ]
        
        If no clear entities are found, return an empty JSON array.
        
        Here is the text to analyze:
        "{text}"
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            if response_content.startswith("```json"):
                response_content = response_content.strip("```json\n").strip("\n```")

            return json.loads(response_content)
        except Exception as e:
            print(f"Error during NER extraction for query: {e}")
            return []

    def create_knowledge_base(self, sentences: List[str]):
        """
        Takes a list of sentences, extracts entities and relationships,
        and dynamically creates/updates the knowledge base in Neo4j.
        """
        print("\n--- Starting Knowledge Base Creation ---")
        for i, sentence in enumerate(sentences):
            print(f"Processing sentence {i+1}: '{sentence}'")
            # Use the specific function for graph creation
            extracted_data = self._extract_entities_for_creation(sentence)
            
            if not extracted_data:
                print("No entities or relationships extracted.")
                continue

            for data in extracted_data:
                # Dynamically construct a Cypher query using MERGE
                # Now using backticks around labels to handle spaces correctly.
                cypher_query = f"""
                MERGE (e1:`{data['entity1_label']}` {{name: $entity1_name}})
                MERGE (e2:`{data['entity2_label']}` {{name: $entity2_name}})
                MERGE (e1)-[:`{data['relationship']}`]->(e2)
                """
                self.neo4j_graph.query(
                    cypher_query,
                    params={
                        "entity1_name": data["entity1"],
                        "entity2_name": data["entity2"]
                    }
                )
            print(f"Successfully created/updated graph for sentence {i+1}.")
        print("--- Knowledge Base Creation Complete ---")

    def retrieve_from_knowledge_base(self, query: str) -> Optional[str]:
        """
        Extracts entities from a query and retrieves relevant context from Neo4j.
        """
        print(f"\n--- Retrieving from Knowledge Base for query: '{query}' ---")
        # Use the specific function for query entity extraction
        extracted_entities = self._extract_entities_for_query(query)
        
        # Get a list of just the entity names from the extracted data.
        query_entity_names = [data['name'] for data in extracted_entities]
        
        if not query_entity_names:
            print("No relevant entities extracted from the query.")
            return None

        # Cypher query to find related nodes and relationships
        cypher_query = """
        MATCH (n)-[r]-(m)
        WHERE n.name IN $entity_names
        RETURN n.name AS source, labels(n) AS source_label, type(r) AS relationship, m.name AS target, labels(m) AS target_label
        """
        
        results = self.neo4j_graph.query(cypher_query, params={"entity_names": query_entity_names})
        
        context_list = []
        for record in results:
            source_node = f"({record['source']}:{':'.join(record['source_label'])})"
            target_node = f"({record['target']}:{':'.join(record['target_label'])})"
            context_list.append(
                f"{source_node} -[:{record['relationship']}]-> {target_node}"
            )
        
        if not context_list:
            print(f"No context found for entities: {query_entity_names}")
            return None
        
        return "\n".join(context_list)

# if __name__ == "__main__":
#     # --- Example Usage ---
    
#     # Instantiate the Neo4j workflow manager
#     workflow = Neo4jWorkflow(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)
    
#     # Sample sentences to build the knowledge base from
#     sample_sentences = document_chunks

#     try:
#         # 1. Create the knowledge base
#         workflow.create_knowledge_base(sample_sentences)
        
#         # 2. Retrieve information from the knowledge base
#         user_query = "Who founded Apple and what is its current CEO?"
#         graph_context = workflow.retrieve_from_knowledge_base(user_query)
        
#         if graph_context:
#             print("\n--- Retrieved Context from Neo4j ---")
#             print(graph_context)
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
