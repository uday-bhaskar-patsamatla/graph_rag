# Chunking Pipeline (New): 

Splits large documents into smaller, semantically coherent units.

## Strategy: 

We'll primarily use LangChain's RecursiveCharacterTextSplitter. This is a robust choice as it attempts to split by common separators (like paragraphs, sentences) before resorting to fixed character counts, preserving context. We'll also consider chunk_overlap to ensure continuity.

## Output: 

A list of text chunks, each associated with metadata (original document ID, chunk index, etc.).

