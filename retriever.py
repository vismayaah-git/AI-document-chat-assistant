import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS


# FAISS: meaning-based vector search
def create_vector_store(chunks, embeddings):
    return FAISS.from_texts(chunks, embeddings)


# BM25: keyword-based search
def create_bm25(chunks):
    return BM25Okapi([chunk.lower().split() for chunk in chunks])


# Hybrid RAG: combine semantic and keyword search
def hybrid_search(chunks, vector_store, bm25, query, k=3):
    vector_results = vector_store.similarity_search(query, k=k)
    query_terms = set(query.lower().split())
    keyword_scores = bm25.get_scores(list(query_terms))

    matching_indices = sorted(
        (i for i, chunk in enumerate(chunks) if query_terms.intersection(chunk.lower().split())),
        key=keyword_scores.__getitem__,
        reverse=True,
    )
    keyword_results = [chunks[i] for i in matching_indices[:k]]

    return vector_results, keyword_results

# Reranking: choose the most relevant chunks
@st.cache_resource
def create_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, chunks, top_k=3):
    if not chunks:
        return []

    model = create_reranker()
    scores = model.predict([(query, chunk) for chunk in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda item: float(item[0]), reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]