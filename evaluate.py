# =============================================================================
# evaluate.py — Cranfield Evaluation: basic TF-IDF vs POS-weighted TF-IDF vs BM25
# =============================================================================

import os
import re
import pickle
import warnings
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE = os.path.dirname(os.path.abspath(__file__))

QUERIES = {
    "Q0": "deep learning image classification",
    "Q1": "neural network optimisation",
    "Q2": "natural language processing transformer",
    "Q3": "reinforcement learning robotics",
    "Q4": "graph neural network node classification",
    "Q5": "convolutional network object detection",
}

TOP_K = 10
_stop_words = set(stopwords.words("english"))
_stemmer = PorterStemmer()


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and t.strip()]
    return " ".join(_stemmer.stem(t) for t in tokens)


def load_qrels(path: str) -> dict:
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, did, grade = parts[0], parts[1], int(parts[2])
            qrels[qid][did] = grade
    return dict(qrels)


def validate_qrels_alignment(qrels_data: dict, available_doc_ids: list) -> tuple[int, int]:
    qrels_doc_ids = {did for rel in qrels_data.values() for did in rel.keys()}
    available_ids = {str(did) for did in available_doc_ids}
    matched = sum(1 for did in qrels_doc_ids if did in available_ids)
    return matched, len(qrels_doc_ids)


def _precision_at_k(ranked_ids: list, relevant: dict, k: int) -> float:
    hits = sum(1 for did in ranked_ids[:k] if relevant.get(did, 0) >= 1)
    return hits / k


def _recall_at_k(ranked_ids: list, relevant: dict, k: int) -> float:
    n_relevant = sum(1 for g in relevant.values() if g >= 1)
    if n_relevant == 0:
        return 0.0
    hits = sum(1 for did in ranked_ids[:k] if relevant.get(did, 0) >= 1)
    return hits / n_relevant


def _average_precision(ranked_ids: list, relevant: dict) -> float:
    n_relevant = sum(1 for g in relevant.values() if g >= 1)
    if n_relevant == 0:
        return 0.0
    hits, running_sum = 0, 0.0
    for rank, did in enumerate(ranked_ids, 1):
        if relevant.get(did, 0) >= 1:
            hits += 1
            running_sum += hits / rank
    return running_sum / n_relevant


def _reciprocal_rank(ranked_ids: list, relevant: dict) -> float:
    for rank, did in enumerate(ranked_ids, 1):
        if relevant.get(did, 0) >= 1:
            return 1.0 / rank
    return 0.0


def _dcg(ranked_ids: list, relevant: dict, k: int) -> float:
    return sum(relevant.get(did, 0) / np.log2(rank + 1) for rank, did in enumerate(ranked_ids[:k], 1))


def _idcg(relevant: dict, k: int) -> float:
    grades = sorted((g for g in relevant.values() if g > 0), reverse=True)
    return sum(g / np.log2(rank + 1) for rank, g in enumerate(grades[:k], 1))


def _ndcg_at_k(ranked_ids: list, relevant: dict, k: int) -> float:
    ideal = _idcg(relevant, k)
    return _dcg(ranked_ids, relevant, k) / ideal if ideal > 0 else 0.0


def evaluate_run(ranked_results: dict, qrels_data: dict, k: int = 10) -> dict:
    rows = []
    for qid, ranked in ranked_results.items():
        rel = qrels_data.get(qid, {})
        rows.append({
            "qid": qid,
            "P@5": _precision_at_k(ranked, rel, 5),
            "P@10": _precision_at_k(ranked, rel, k),
            "Recall@10": _recall_at_k(ranked, rel, k),
            "AP": _average_precision(ranked, rel),
            "RR": _reciprocal_rank(ranked, rel),
            "NDCG@10": _ndcg_at_k(ranked, rel, k),
        })

    means = {
        "qid": "MEAN",
        "P@5": np.mean([r["P@5"] for r in rows]),
        "P@10": np.mean([r["P@10"] for r in rows]),
        "Recall@10": np.mean([r["Recall@10"] for r in rows]),
        "AP": np.mean([r["AP"] for r in rows]),
        "RR": np.mean([r["RR"] for r in rows]),
        "NDCG@10": np.mean([r["NDCG@10"] for r in rows]),
    }
    return {"rows": rows, "means": means}


def run_tfidf_base(queries: dict, vectorizer, tfidf_base_matrix, doc_ids: list, k: int = 10) -> dict:
    results = {}
    for qid, q in queries.items():
        qv = vectorizer.transform([_preprocess(q)])
        sco = cosine_similarity(qv, tfidf_base_matrix).flatten()
        top = sco.argsort()[::-1][:k]
        results[qid] = [str(doc_ids[i]) for i in top]
    return results


def run_tfidf_pos(queries: dict, vectorizer, tfidf_pos_matrix, pos_weight_diag, doc_ids: list, k: int = 10) -> dict:
    results = {}
    for qid, q in queries.items():
        qv = vectorizer.transform([_preprocess(q)]).dot(pos_weight_diag)
        sco = cosine_similarity(qv, tfidf_pos_matrix).flatten()
        top = sco.argsort()[::-1][:k]
        results[qid] = [str(doc_ids[i]) for i in top]
    return results


def run_bm25(queries: dict, tokenised_docs: list, doc_ids: list, k: int = 10) -> dict:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenised_docs)
    results = {}
    for qid, q in queries.items():
        q_tokens = _preprocess(q).split()
        scores = bm25.get_scores(q_tokens)
        top = scores.argsort()[::-1][:k]
        results[qid] = [str(doc_ids[i]) for i in top]
    return results


def print_table(label: str, eval_data: dict, metrics: list) -> None:
    col_w = 10
    header = f"{'Query':<8}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(f"\n--- {label} ---")
    print(header)
    print("-" * len(header))
    for row in eval_data["rows"]:
        print(f"{row['qid']:<8}" + "".join(f"{row[m]:>{col_w}.4f}" for m in metrics))
    print("-" * len(header))
    means = eval_data["means"]
    print(f"{'MEAN':<8}" + "".join(f"{means[m]:>{col_w}.4f}" for m in metrics))


if __name__ == "__main__":
    print("Loading artifacts...")
    with open(f"{BASE}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{BASE}/pos_weight_vector.pkl", "rb") as f:
        pos_weight_vector = pickle.load(f)
    with open(f"{BASE}/doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)
    with open(f"{BASE}/doc_titles.pkl", "rb") as f:
        doc_titles = pickle.load(f)
    with open(f"{BASE}/doc_abstracts.pkl", "rb") as f:
        doc_abstracts = pickle.load(f)
    tfidf_base_matrix = sp.load_npz(f"{BASE}/tfidf_base_matrix.npz")
    tfidf_pos_matrix = sp.load_npz(f"{BASE}/tfidf_pos_matrix.npz")
    pos_weight_diag = sp.diags(pos_weight_vector)
    print(f"Loaded {len(doc_ids):,} documents")

    qrels = load_qrels(f"{BASE}/qrels.tsv")
    matched, total = validate_qrels_alignment(qrels, doc_ids)
    print(f"Qrels alignment: {matched}/{total} doc IDs found in corpus")

    print("Preprocessing corpus for BM25...")
    combined_texts = [
        str(doc_titles[i]) * 2 + " " + str(doc_abstracts[i])
        for i in range(len(doc_ids))
    ]
    tokenised_docs = [_preprocess(t).split() for t in combined_texts]

    metrics = ["P@5", "P@10", "Recall@10", "AP", "RR", "NDCG@10"]

    print("\nRunning basic TF-IDF (no POS weighting)...")
    base_results = run_tfidf_base(QUERIES, vectorizer, tfidf_base_matrix, doc_ids, k=TOP_K)
    base_eval = evaluate_run(base_results, qrels)

    print("Running POS-weighted TF-IDF...")
    pos_results = run_tfidf_pos(QUERIES, vectorizer, tfidf_pos_matrix, pos_weight_diag, doc_ids, k=TOP_K)
    pos_eval = evaluate_run(pos_results, qrels)

    print("Running BM25...")
    bm25_results = run_bm25(QUERIES, tokenised_docs, doc_ids, k=TOP_K)
    bm25_eval = evaluate_run(bm25_results, qrels)

    print_table("Basic TF-IDF (no POS weighting)", base_eval, metrics)
    print_table("POS-weighted TF-IDF", pos_eval, metrics)
    print_table("BM25 (rank-bm25, Okapi)", bm25_eval, metrics)

    print("\n=== Summary: Three-System Comparison ===")
    print(f"{'Metric':<14} {'TF-IDF Base':>14} {'TF-IDF POS':>14} {'BM25':>10}")
    print("-" * 54)
    for metric in metrics:
        bv = base_eval["means"][metric]
        pv = pos_eval["means"][metric]
        mv = bm25_eval["means"][metric]
        print(f"{metric:<14} {bv:>14.4f} {pv:>14.4f} {mv:>10.4f}")
