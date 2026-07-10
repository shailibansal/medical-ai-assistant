from Bio import Entrez

Entrez.email = "shailib23@gmail.com"


def search_pubmed(query, max_results=5):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )

    results = Entrez.read(handle)
    handle.close()

    return results["IdList"]


def fetch_papers(id_list):

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
                article_info["Abstract"]["AbstractText"]
            )

        journal = article_info["Journal"]["Title"]

        authors = []

        if "AuthorList" in article_info:

            for author in article_info["AuthorList"]:

                if "LastName" in author and "ForeName" in author:
                    authors.append(
                        author["ForeName"] + " " + author["LastName"]
                    )

        papers.append({
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": authors
        })

    return papers