from services.vectordb import search_papers
import time
from services.reranker import rerank_documents

# --------------------------------------------------
# Evaluation Dataset
# --------------------------------------------------

EVALUATION_DATASET = [

    {
        "query": "What are the molecular mechanisms and pathology of Alzheimer's disease?",
        "relevant_pmids": [
            "38733347",
            "39236855"
        ]
    },

    {
        "query": "What are the pathogenesis and therapeutic approaches for Alzheimer's disease?",
        "relevant_pmids": [
            "39236855",
            "38733347"
        ]
    },

    {
        "query": "What are animal models used for studying Alzheimer's disease?",
        "relevant_pmids": [
            "36317468"
        ]
    },

    {
        "query": "How are Alzheimer's disease animal models used to study disease mechanisms?",
        "relevant_pmids": [
            "36317468"
        ]
    },

    {
        "query": "What are the complications of metabolic surgery for type 2 diabetes?",
        "relevant_pmids": [
            "42460326"
        ]
    },

    {
        "query": "What surgical techniques are used for metabolic treatment of type 2 diabetes?",
        "relevant_pmids": [
            "42460326"
        ]
    },

    {
        "query": "What are the molecular mechanisms involved in diabetic retinopathy?",
        "relevant_pmids": [
            "42460327"
        ]
    },

    {
        "query": "How does neurovascular unit dysfunction contribute to diabetic retinopathy?",
        "relevant_pmids": [
            "42460327"
        ]
    },

    {
        "query": "What factors contribute to type 2 diabetes remission?",
        "relevant_pmids": [
            "42460553"
        ]
    },

    {
        "query": "How is remission-oriented care implemented in type 2 diabetes?",
        "relevant_pmids": [
            "42460553"
        ]
    },

    {
        "query": "What are the causes of kidney disease in patients with diabetes?",
        "relevant_pmids": [
            "42462258"
        ]
    },

    {
        "query": "What findings can be observed in kidney biopsies of patients with diabetes?",
        "relevant_pmids": [
            "42462258"
        ]
    },

    {
        "query": "How does monk fruit extract affect glycemic control?",
        "relevant_pmids": [
            "42460423"
        ]
    },

    {
        "query": "What are the effects of monk fruit extract on glucose metabolism and body weight?",
        "relevant_pmids": [
            "42460423"
        ]
    },

    {
        "query": "What are the metabolic effects of non-nutritive sweeteners in obese mice?",
        "relevant_pmids": [
            "42460423"
        ]
    }
]


# --------------------------------------------------
# Precision@K
# --------------------------------------------------

def precision_at_k(retrieved_pmids, relevant_pmids, k):

    retrieved = retrieved_pmids[:k]

    relevant_count = sum(
        1
        for pmid in retrieved
        if pmid in relevant_pmids
    )

    return relevant_count / k


# --------------------------------------------------
# Recall@K
# --------------------------------------------------

def recall_at_k(retrieved_pmids, relevant_pmids, k):

    retrieved = retrieved_pmids[:k]

    relevant_count = sum(
        1
        for pmid in retrieved
        if pmid in relevant_pmids
    )

    if len(relevant_pmids) == 0:
        return 0

    return relevant_count / len(relevant_pmids)


# --------------------------------------------------
# Mean Reciprocal Rank
# --------------------------------------------------

def reciprocal_rank(retrieved_pmids, relevant_pmids):

    for rank, pmid in enumerate(retrieved_pmids, start=1):

        if pmid in relevant_pmids:
            return 1 / rank

    return 0


# --------------------------------------------------
# Evaluate one query
# --------------------------------------------------

def evaluate_query(query, relevant_pmids, k=5):

    start_time = time.perf_counter()

    results = search_papers(
        query,
        n_results=k
    )

    results = rerank_documents(
    query,
    results,
    top_k=5
)

    end_time = time.perf_counter()

    latency = end_time - start_time

    retrieved_pmids = [
        str(result["pmid"])
        for result in results
    ]

    precision = precision_at_k(
        retrieved_pmids,
        relevant_pmids,
        k
    )

    recall = recall_at_k(
        retrieved_pmids,
        relevant_pmids,
        k
    )

    mrr = reciprocal_rank(
        retrieved_pmids,
        relevant_pmids
    )

    return {
        "query": query,
        "retrieved_pmids": retrieved_pmids,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "latency_seconds": latency
    }


# --------------------------------------------------
# Run complete evaluation
# --------------------------------------------------

def run_evaluation():

    print("\n")
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    results = []

    for item in EVALUATION_DATASET:

        result = evaluate_query(
            item["query"],
            item["relevant_pmids"],
            k=5
        )

        results.append(result)

        print("\nQuery:")
        print(item["query"])

        print("\nRetrieved PMIDs:")
        print(result["retrieved_pmids"])

        print(
            f"Precision@5: "
            f"{result['precision_at_k']:.2f}"
        )

        print(
            f"Recall@5: "
            f"{result['recall_at_k']:.2f}"
        )

        print(
            f"MRR: "
            f"{result['mrr']:.2f}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']:.4f} seconds"
        )

        print("-" * 70)

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------

    avg_precision = sum(
        r["precision_at_k"]
        for r in results
    ) / len(results)

    avg_recall = sum(
        r["recall_at_k"]
        for r in results
    ) / len(results)

    avg_mrr = sum(
        r["mrr"]
        for r in results
    ) / len(results)

    avg_latency = sum(
        r["latency_seconds"]
        for r in results
    ) / len(results)

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Average Precision@5: {avg_precision:.2f}"
    )

    print(
        f"Average Recall@5:    {avg_recall:.2f}"
    )

    print(
        f"Average MRR:          {avg_mrr:.2f}"
    )

    print(
        f"Average Latency:      {avg_latency:.4f} seconds"
    )

    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()