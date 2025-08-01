from services.data_retrieval import HybridRetrieval
retrieval_pipeline = HybridRetrieval()

# Example queries
query1 = "Who founded Apple Inc.?"
query2 = "What company is Tim Cook the CEO of?"
query3 = "What is Pixar?"

# --- Process Query 1 ---
print(f"*** Processing Query: '{query1}' ***")
retrieved_context1 = retrieval_pipeline.retrieve_context(query1)
final_answer1 = retrieval_pipeline.generate_answer(query1, retrieved_context1)
print("\nFinal Answer:")
print(final_answer1)
print("\n" + "="*50 + "\n")

# --- Process Query 2 ---
print(f"*** Processing Query: '{query2}' ***")
retrieved_context2 = retrieval_pipeline.retrieve_context(query2)
final_answer2 = retrieval_pipeline.generate_answer(query2, retrieved_context2)
print("\nFinal Answer:")
print(final_answer2)
print("\n" + "="*50 + "\n")

# --- Process Query 3 ---
print(f"*** Processing Query: '{query3}' ***")
retrieved_context3 = retrieval_pipeline.retrieve_context(query3)
final_answer3 = retrieval_pipeline.generate_answer(query3, retrieved_context3)
print("\nFinal Answer:")
print(final_answer3)
print("\n" + "="*50 + "\n")
