from retriever import FinancialRetriever


retriever = FinancialRetriever()

results = retriever.search(
    "Tüpraş'ın 2025 yılında toplam geliri ne kadardı?"
)

for i, result in enumerate(results, 1):
    print(f"\n--- RESULT {i} ---")
    print(result["reranker_score"])
    print(result["metadata"])
    print(result["content"][:500])