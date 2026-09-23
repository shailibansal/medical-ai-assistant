import re
from services.vectordb import search_papers, add_papers
from services.pubmed import search_pubmed, fetch_papers
from services.llm import ask_llm
from services.reranker import rerank_documents


MAX_ACCEPTABLE_DISTANCE = 1.30
MIN_RELEVANT_PAPERS = 2


def enough_relevant_evidence(papers):

    if not papers:
        return False

    best_distance = papers[0]["distance"]

    relevant_count = sum(
        1
        for paper in papers
        if paper["distance"] <= MAX_ACCEPTABLE_DISTANCE
    )

    if (
        best_distance <= MAX_ACCEPTABLE_DISTANCE
        and relevant_count >= MIN_RELEVANT_PAPERS
    ):
        return True

    return False


def build_context(papers):

    context = ""

    for i, paper in enumerate(papers):

        context += f"""
SOURCE {i + 1}

PMID: {paper['pmid']}

Title:
{paper['title']}

Journal:
{paper['journal']}

Authors:
{paper['authors']}

Content:
{paper['content']}

------------------------------------
"""

    return context


def build_sources(papers):

    sources = []

    for paper in papers:

        sources.append({
            "pmid": paper["pmid"],
            "title": paper["title"],
            "journal": paper["journal"],
            "authors": paper["authors"]
        })

    return sources

def validate_citations(answer, papers):

    retrieved_pmids = {
        str(paper["pmid"])
        for paper in papers
    }

    cited_pmids = set(
        re.findall(
            r"(?:\[|\()PMID:\s*(\d+)(?:\]|\))",
            answer
        )
    )

    valid_citations = cited_pmids.intersection(
        retrieved_pmids
    )

    invalid_citations = cited_pmids.difference(
        retrieved_pmids
    )

    return {
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "citation_count": len(cited_pmids),
        "all_citations_valid": len(invalid_citations) == 0
    }



def answer_question(question):

    used_pubmed_fallback = False

    # --------------------------------------------------
    # STEP 1: Search local ChromaDB
    # --------------------------------------------------

    print("\n======================================")
    print("LOCAL VECTOR SEARCH")
    print("======================================")

    papers = search_papers(
        question,
        n_results=10
    )

    papers = rerank_documents(
    question,
    papers,
    top_k=5)

    # --------------------------------------------------
    # STEP 2: Check local evidence
    # --------------------------------------------------

    evidence_available = enough_relevant_evidence(papers)

    if evidence_available:

        print("Local evidence is sufficient.")

        selected_papers = papers[:3]

        context = build_context(selected_papers)

        answer = ask_llm(
            context,
            question
        )

        citation_validation = validate_citations(
        answer,
        selected_papers
        )

        return {
            "question": question,
            "answer": answer,
            "sources": build_sources(selected_papers),
            "citation_validation": citation_validation,
            "retrieval_status": "local_evidence"
        }

    # --------------------------------------------------
    # STEP 3: Local evidence insufficient
    # --------------------------------------------------

    print("Local evidence insufficient.")
    print("Searching PubMed for recent papers...")

    used_pubmed_fallback = True

    # --------------------------------------------------
    # STEP 4: Search PubMed
    # --------------------------------------------------

    ids = search_pubmed(
        question,
        max_results=5,
        recent_years=5
    )

    if not ids:

        return {
            "question": question,
            "answer": "I could not find relevant recent research papers.",
            "sources": [],
            "retrieval_status": "pubmed_no_results"
        }

    print(f"PubMed returned {len(ids)} papers.")

    # --------------------------------------------------
    # STEP 5: Fetch papers
    # --------------------------------------------------

    new_papers = fetch_papers(ids)

    if not new_papers:

        return {
            "question": question,
            "answer": "I could not retrieve the PubMed research papers.",
            "sources": [],
            "retrieval_status": "pubmed_fetch_failed"
        }

    print(f"Fetched {len(new_papers)} papers.")

    # --------------------------------------------------
    # STEP 6: Store papers
    # --------------------------------------------------

    add_papers(new_papers)

    print("New papers stored in ChromaDB.")

    # --------------------------------------------------
    # STEP 7: Search again
    # --------------------------------------------------

    papers = search_papers(
        question,
        n_results=5
    )

    papers = rerank_documents(
            question,
            papers,
            top_k=5
)

    evidence_available = enough_relevant_evidence(papers)

    # --------------------------------------------------
    # STEP 8: Check evidence again
    # --------------------------------------------------

    if not evidence_available:

        return {
            "question": question,
            "answer": "I found recent papers, but they did not provide enough relevant evidence to answer the question reliably.",
            "sources": [],
            "retrieval_status": "insufficient_pubmed_evidence"
        }

    # --------------------------------------------------
    # STEP 9: Generate answer
    # --------------------------------------------------

    selected_papers = papers[:3]

    context = build_context(selected_papers)

    answer = ask_llm(
        context,
        question
    )

    citation_validation = validate_citations(
    answer,
    selected_papers
    )

    # --------------------------------------------------
    # STEP 10: Return result
    # --------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": build_sources(selected_papers),
        "citation_validation": citation_validation,
        "retrieval_status": (
            "pubmed_fallback"
            if used_pubmed_fallback
            else "local_evidence"
        )
}