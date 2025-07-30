# NER & Relationship Extraction (New): 

Leverages OpenAI LLMs orchestrated by LangChain to identify entities (persons, organizations, concepts, etc.) and the relationships between them within each chunk.

## LLM: 

OpenAI's gpt-4o or gpt-4o-mini (for cost-efficiency during development/initial ingestion) will be used for their strong instruction following and JSON output capabilities.

## Schema:

We'll define a clear JSON schema (or Pydantic model) for the expected entities and relationships, guiding the LLM's output.

## Prompt Engineering: 

Crucial for quality. We'll use system prompts, clear instructions, and few-shot examples to maximize extraction accuracy.