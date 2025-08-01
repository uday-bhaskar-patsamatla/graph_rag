import os
from dotenv import load_dotenv
from utils.neo4j_db import Neo4jWorkflow
from utils.chroma_db import retrieve_from_chroma


from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
load_dotenv()

# Get the API keys and database details from environment variables.
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

class HybridRetrieval:
    """
    A class to demonstrate a hybrid retrieval pipeline using both a Neo4j knowledge graph
    for structured facts and a ChromaDB vector store for semantic search.
    """
    def __init__(self):
        """
        Initializes the Neo4j workflow, ChromaDB collection, and OpenAI client.
        """
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize the embedding model
        # embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        


    def retrieve_context(self, user_query: str) -> str:
        """
        Performs a hybrid retrieval, combining results from Neo4j and ChromaDB.
        """
        retrieved_context = []
        
        # 1. Retrieve structured context from Neo4j
        print("\nSearching Neo4j knowledge graph...")
        workflow = Neo4jWorkflow(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)
        graph_context = workflow.retrieve_from_knowledge_base(user_query)
        

        if graph_context:
            retrieved_context.append("--- Context from Knowledge Graph (Neo4j) ---\n" + graph_context)
            
        # 2. Retrieve unstructured context from ChromaDB
        print("Searching ChromaDB vector store...")
        vector_results = retrieve_from_chroma(user_query, k=2)
        vector_context = "\n".join([doc.page_content for doc in vector_results])
        if vector_context:
            retrieved_context.append("--- Context from Vector Store (ChromaDB) ---\n" + vector_context)

        return "\n\n".join(retrieved_context) if retrieved_context else "No relevant context found."

    def generate_answer(self, user_query: str, context: str) -> str:
        """
        Generates a final answer using the retrieved context and the user's query.
        """
        if context == "No relevant context found.":
            return "I'm sorry, I couldn't find enough information to answer that question."

        prompt = f"""
        Given the following context, answer the user's question.
        If the context does not contain the answer, state that you cannot find the answer.
        
        Context:
        {context}
        
        Question:
        {user_query}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"

