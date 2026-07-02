from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    text = text.lower()
    return re.findall(r"\w+", text)


def build_document_text(item):
    return f"""
    {item['name']}
    {item['description']}
    {' '.join(item['keys'])}
    {' '.join(item['job_levels'])}
    """

def bm25_search(query, candidates):
    docs = [tokenize(build_document_text(x)) for x in candidates]

    bm25 = BM25Okapi(docs)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    return scores