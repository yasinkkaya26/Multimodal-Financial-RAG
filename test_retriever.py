from retriever import FinancialRetriever


retriever = FinancialRetriever()

results = retriever.search(
    "Tüpraş'ın 2025 yılında toplam geliri ne kadardı?"
)

for i, result in enumerate(results, 1):
    print(f"\n--- RESULT {i} ---")
    print(f"Reranker: {result['reranker_score']:.4f}")
    print(f"Vector rank: {result['vector_rank']}")
    print(f"Text rank: {result['text_rank']}")
    print(f"RRF score: {result['rrf_score']:.6f}")
    print(f"Metadata: {result['metadata']}")
    print(result["content"][:500])