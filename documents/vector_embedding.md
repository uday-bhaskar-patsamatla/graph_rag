# Vector Embedding Generation (New): 

Creates numerical representations (embeddings) of the document chunks.

## OpenAI Embeddings: 

Using langchain_openai.OpenAIEmbeddings with a cost-effective model like text-embedding-3-small.

## Vector Store Population (New): 

Stores the chunk embeddings in ChromaDB.

## ChromaDB: 

The chosen vector database for efficient similarity search.

## LangChain Chroma integration: 

Simplifies adding and querying embeddings. For production, we'll ensure ChromaDB runs in client-server mode with persistent storage.