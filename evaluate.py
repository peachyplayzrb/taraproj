# =============================================================================
# evaluate.py — Cranfield Evaluation: TF-IDF cosine vs BM25 baseline
# =============================================================================
# Runs both retrieval models against the 6 standard test queries and the
# graded relevance judgments in qrels.tsv, then prints a side-by-side metric
# table covering P@5, P@10, Recall@10, MAP, MRR, NDCG@10.
#
# Usage (with venv active):
#   pip install rank-bm25          # only needed once
#   python evaluate.py
#
# Artifacts required (in same directory):
#   vectorizer.pkl, pos_weight_vector.pkl, doc_ids.pkl,
#   doc_titles.pkl, doc_abstracts.pkl, tfidf_pos_matrix.npz
# =============================================================================

import os, re, pickle, warnings
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Queries (same 6 used throughout the notebook)
# ---------------------------------------------------------------------------
QUERIES = {
    "Q0": "deep learning image classification",
    "Q1": "neural network optimisation",
    "Q2": "natural language processing transformer",
    "Q3": "reinforcement learning robotics",
    "Q4": "graph neural network node classification",
    "Q5": "convolutional network object detection",
}

TOP_K = 10   # rank cut-off for @10 metrics

# ---------------------------------------------------------------------------
# Preprocessing (must match notebook indexing pipeline exactly)
# ---------------------------------------------------------------------------
_stop_words = set(stopwords.words("english"))
_stemmer    = PorterStemmer()


def _preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^\w\s\-]", "", text)
    text   = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and t.strip()]
    return " ".join(_stemmer.stem(t) for t in tokens)


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
print("Loading artifacts …")
with open(f"{BASE}/vectorizer.pkl",        "rb") as f: vectorizer        = pickle.load(f)
with open(f"{BASE}/pos_weight_vector.pkl", "rb") as f: pos_weight_vector = pickle.load(f)
with open(f"{BASE}/doc_ids.pkl",           "rb") as f: doc_ids           = pickle.load(f)
with open(f"{BASE}/doc_titles.pkl",        "rb") as f: doc_titles        = pickle.load(f)
with open(f"{BASE}/doc_abstracts.pkl",     "rb") as f: doc_abstracts     = pickle.load(f)
tfidf_pos_matrix = sp.load_npz(f"{BASE}/tfidf_pos_matrix.npz")
pos_weight_diag  = sp.diags(pos_weight_vector)
print(f"Loaded {len(doc_ids):,} documents.")

# Preprocess all documents once for BM25
print("Preprocessing documents for BM25 (this may take a moment) …")
combined_texts = [
    str(doc_titles[i]) * 2 + " " + str(doc_abstracts[i])
    for i in range(len(doc_ids))
]
tokenised_docs = [_preprocess(t).split() for t in combined_texts]
print("Preprocessing done.")

# ---------------------------------------------------------------------------
# Load qrels
# ---------------------------------------------------------------------------
def load_qrels(path: str) -> dict:
    """Return {query_id: {doc_id_str: grade}} from a tab-separated qrels file."""
    qrels: dict = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, did, grade = parts[0], parts[1], int(parts[2])
            qrels[qid][did] = grade
    return dict(qrels)


qrels = load_qrels(f"{BASE}/qrels.tsv")

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

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
    total = 0.0
    for rank, did in enumerate(ranked_ids[:k], 1):
        gain = relevant.get(did, 0)
        total += gain / np.log2(rank + 1)
    return total


def _idcg(relevant: dict, k: int) -> float:
    grades = sorted((g for g in relevant.values() if g > 0), reverse=True)
    return sum(g / np.log2(rank + 1) for rank, g in enumerate(grades[:k], 1))


def _ndcg_at_k(ranked_ids: list, relevant: dict, k: int) -> float:
    ideal = _idcg(relevant, k)
    return _dcg(ranked_ids, relevant, k) / ideal if ideal > 0 else 0.0


def evaluate_run(ranked_results: dict, qrels_data: dict, k: int = 10) -> dict:
    """
    Compute per-query metrics and macro-averages.

    Args:
        ranked_results: {query_id: [doc_id_str, ...]}  (ordered top-k)
        qrels_data:     {query_id: {doc_id_str: grade}}
        k:              rank cut-off

    Returns:
        dict with per-query rows and means.
    """
    rows = []
    for qid, ranked in ranked_results.items():
        rel = qrels_data.get(qid, {})
        rows.append({
            "qid":         qid,
            "P@5":         _precision_at_k(ranked, rel, 5),
            "P@10":        _precision_at_k(ranked, rel, k),
            "Recall@10":   _recall_at_k(ranked, rel, k),
            "AP":          _average_precision(ranked, rel),
            "RR":          _reciprocal_rank(ranked, rel),
            "NDCG@10":     _ndcg_at_k(ranked, rel, k),
        })
    means = {
        "qid":       "MEAN",
        "P@5":       np.mean([r["P@5"]       for r in rows]),
        "P@10":      np.mean([r["P@10"]      for r in rows]),
        "Recall@10": np.mean([r["Recall@10"] for r in rows]),
        "AP":        np.mean([r["AP"]        for r in rows]),
        "RR":        np.mean([r["RR"]        for r in rows]),
        "NDCG@10":   np.mean([r["NDCG@10"]  for r in rows]),
    }
    return {"rows": rows, "means": means}


# ---------------------------------------------------------------------------
# TF-IDF cosine retrieval
# ---------------------------------------------------------------------------
def run_tfidf(queries: dict, k: int = 10) -> dict:
    results = {}
    for qid, q in queries.items():
        qv  = vectorizer.transform([_preprocess(q)]).dot(pos_weight_diag)
        sco = cosine_similarity(qv, tfidf_pos_matrix).flatten()
        top = sco.argsort()[::-1][:k]
        results[qid] = [str(doc_ids[i]) for i in top]
    return results


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------
def run_bm25(queries: dict, k: int = 10) -> dict:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank-bm25 not installed. Run: pip install rank-bm25")
        return {qid: [] for qid in queries}

    bm25 = BM25Okapi(tokenised_docs)
    results = {}
    for qid, q in queries.items():
        q_tokens = _preprocess(q).split()
        scores   = bm25.get_scores(q_tokens)
        top      = scores.argsort()[::-1][:k]
        results[qid] = [str(doc_ids[i]) for i in top]
    return results


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
print("\nRunning TF-IDF cosine …")
tfidf_results = run_tfidf(QUERIES, k=TOP_K)
tfidf_eval    = evaluate_run(tfidf_results, qrels)

print("Running BM25 …")
bm25_results = run_bm25(QUERIES, k=TOP_K)
bm25_eval    = evaluate_run(bm25_results, qrels)

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
METRICS = ["P@5", "P@10", "Recall@10", "AP", "RR", "NDCG@10"]
COL_W   = 10

def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_table(label: str, eval_data: dict) -> None:
    header = f"{'Query':<8}" + "".join(f"{m:>{COL_W}}" for m in METRICS)
    print(f"\n--- {label} ---")
    print(header)
    print("-" * len(header))
    for row in eval_data["rows"]:
        print(f"{row['qid']:<8}" + "".join(f"{_fmt(row[m]):>{COL_W}}" for m in METRICS))
    print("-" * len(header))
    m = eval_data["means"]
    print(f"{'MEAN':<8}" + "".join(f"{_fmt(m[k]):>{COL_W}}" for k in METRICS))


print_table("TF-IDF Cosine (POS-weighted)", tfidf_eval)
print_table("BM25 (rank-bm25, Okapi)", bm25_eval)

# ---------------------------------------------------------------------------
# Side-by-side summary (MAP / MRR / NDCG)
# ---------------------------------------------------------------------------
tm = tfidf_eval["means"]
bm = bm25_eval["means"]
print("\n=== Summary: TF-IDF cosine vs BM25 ===")
print(f"{'Metric':<14} {'TF-IDF':>10} {'BM25':>10} {'Delta':>10}")
print("-" * 46)
for metric in METRICS:
    tv, bv = tm[metric], bm[metric]
    print(f"{metric:<14} {tv:>10.4f} {bv:>10.4f} {bv - tv:>+10.4f}")
