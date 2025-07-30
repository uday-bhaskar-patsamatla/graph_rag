User Query
    |
    v
Application Frontend/API Gateway
    |
    v
Query Understanding & Orchestration Layer (e.g., LangChain Agents)
    |
    +------------------------------------------------------------+
    |                                                            |
    v                                                            v
Knowledge Graph Retrieval Module                  Vector Search Module (Optional, for hybrid RAG)
    | (Cypher, Gremlin, etc.)                            | (Embeddings)
    v                                                    v
Knowledge Graph Database                         Vector Database
    | (Entities, Relationships, Properties)              | (Text Chunks, Embeddings, Metadata)
    |                                                    |
    +------------------------------------------------------------+
    |
    v
Context Aggregation & Prompt Construction
    |
    v
Large Language Model (LLM) - (e.g., Gemini, GPT-4)
    |
    v
Response Generation
    |
    v
Application Frontend/User