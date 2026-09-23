from Bio import Entrez

Entrez.email = "shailib23@gmail.com"
Entrez.timeout = 10


def search_pubmed(query, max_results=5, recent_years=5):
    """
    Search PubMed for papers related to the query.

    Only papers from the recent_years window are retrieved.
    """

    try:
        # PubMed date filter
        date_filter = f'("last {recent_years} years"[dp])'

        search_term = f"({query}) AND {date_filter}"

        handle = Entrez.esearch(
            db="pubmed",
            term=search_term,
            retmax=max_results,
            sort="relevance"
        )

        results = Entrez.read(handle)
        handle.close()

        return results["IdList"]

    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def fetch_papers(id_list):
    if not id_list:
        return []

    try:
        ids = ",".join(id_list)

        handle = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="abstract",
            retmode="xml"
        )

        records = Entrez.read(handle)
        handle.close()

        papers = []
        articles = records["PubmedArticle"]

        for article in articles:

            citation = article["MedlineCitation"]
            article_info = citation["Article"]

            title = article_info.get("ArticleTitle", "")

            abstract = ""

            if "Abstract" in article_info:
                abstract = " ".join(
                    str(text)
                    for text in article_info["Abstract"]["AbstractText"]
                )

            journal = article_info.get(
                "Journal", {}
            ).get("Title", "")

            authors = []

            if "AuthorList" in article_info:

                for author in article_info["AuthorList"]:

                    if (
                        "LastName" in author
                        and "ForeName" in author
                    ):
                        authors.append(
                            author["ForeName"]
                            + " "
                            + author["LastName"]
                        )

            pmid = citation["PMID"]

            papers.append({
                "pmid": str(pmid),
                "title": str(title),
                "abstract": abstract,
                "journal": str(journal),
                "authors": authors
            })

        return papers

    except Exception as e:
        print(f"PubMed fetch error: {e}")
        return []