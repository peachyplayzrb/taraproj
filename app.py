from flask import Flask, render_template, request, jsonify
import pickle
import scipy.sparse as sp
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ── arXiv ID helpers ─────────────────────────────────────────
def float_to_arxiv_id(doc_id):
    """Convert float like 704.0033 to proper arXiv ID string 0704.0033"""
    raw   = str(doc_id)
    parts = raw.split('.')
    left  = parts[0].zfill(4)
    right = parts[1] if len(parts) > 1 else '0000'
    return f'{left}.{right}'

def arxiv_url(doc_id):
    return f'https://arxiv.org/abs/{float_to_arxiv_id(doc_id)}'

# ── Load index ───────────────────────────────────────────────
def load_index():
    with open(f'{BASE}/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'{BASE}/pos_weight_vector.pkl', 'rb') as f:
        pos_weight_vector = pickle.load(f)
    with open(f'{BASE}/doc_ids.pkl', 'rb') as f:
        doc_ids = pickle.load(f)
    with open(f'{BASE}/doc_titles.pkl', 'rb') as f:
        doc_titles = pickle.load(f)
    with open(f'{BASE}/doc_abstracts.pkl', 'rb') as f:
        doc_abstracts = pickle.load(f)
    tfidf_matrix     = sp.load_npz(f'{BASE}/tfidf_base_matrix.npz')
    tfidf_pos_matrix = sp.load_npz(f'{BASE}/tfidf_pos_matrix.npz')

    try:
        with open(f'{BASE}/click_store.pkl', 'rb') as f:
            raw = pickle.load(f)
        click_store = defaultdict(lambda: defaultdict(float))
        for k, v in raw.items():
            for doc_id, count in v.items():
                click_store[k][float(doc_id)] += count
    except Exception:
        click_store = defaultdict(lambda: defaultdict(float))

    # Parse year from arXiv ID (YYMM.XXXXX format)
    doc_years = []
    for doc_id in doc_ids:
        try:
            prefix = str(doc_id).split('.')[0].zfill(4)
            yr = int(prefix[:2])
            doc_years.append(2000 + yr if yr < 50 else 1900 + yr)
        except Exception:
            doc_years.append(0)

    # Global click popularity across ALL queries
    global_clicks = defaultdict(float)
    for query_clicks in click_store.values():
        for doc_id, count in query_clicks.items():
            global_clicks[float(doc_id)] += count

    pos_weight_diag = sp.diags(pos_weight_vector)
    return (vectorizer, pos_weight_diag, doc_ids, doc_titles,
            doc_abstracts, tfidf_matrix, tfidf_pos_matrix,
            click_store, doc_years, global_clicks)

print("Loading index...")
(vectorizer, pos_weight_diag, doc_ids, doc_titles,
 doc_abstracts, tfidf_matrix, tfidf_pos_matrix,
 click_store, doc_years, global_clicks) = load_index()

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
print(f"Index loaded — {len(doc_ids)} documents ready")

# ── NLP helpers ──────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\-]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.strip()]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def normalise_query(query):
    return ' '.join(sorted(preprocess(query).split()))

def get_click_score(query, doc_id):
    key    = normalise_query(query)
    clicks = click_store[key]
    if not clicks:
        return 0.0
    fid = float(doc_id)
    if fid not in clicks:
        return 0.0
    return clicks[fid] / max(clicks.values())

def get_global_popularity(doc_id):
    if not global_clicks:
        return 0.0
    fid = float(doc_id)
    return global_clicks.get(fid, 0.0) / max(global_clicks.values())

# ── Search ───────────────────────────────────────────────────
def search(query, top_k=10, sort_by='blended', alpha=0.7):
    processed     = preprocess(query)
    query_vec     = vectorizer.transform([processed])
    query_vec_pos = query_vec.dot(pos_weight_diag)

    cosine_scores  = cosine_similarity(query_vec_pos, tfidf_pos_matrix).flatten()
    click_scores   = np.array([get_click_score(query, doc_ids[i]) for i in range(len(doc_ids))])
    blended_scores = alpha * cosine_scores + (1 - alpha) * click_scores
    year_scores    = np.array(doc_years, dtype=float)
    popularity     = np.array([get_global_popularity(doc_ids[i]) for i in range(len(doc_ids))])

    if sort_by == 'relevance':
        scores = cosine_scores
    elif sort_by == 'blended':
        scores = blended_scores
    elif sort_by == 'popularity':
        scores = np.where(cosine_scores > 0.05, popularity, 0)
    elif sort_by == 'newest':
        scores = np.where(cosine_scores > 0.05, year_scores, 0)
    elif sort_by == 'oldest':
        max_year = float(max(doc_years)) if doc_years else 2026.0
        scores = np.where(cosine_scores > 0.05, max_year - year_scores, 0)
    elif sort_by == 'trending':
        scores = 0.6 * cosine_scores + 0.4 * popularity
    else:
        scores = blended_scores

    top_indices = scores.argsort()[::-1][:top_k]
    q_key = normalise_query(query)

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id   = doc_ids[idx]
        raw      = str(doc_id)
        parts    = raw.split('.')
        left     = parts[0].zfill(4)
        right    = parts[1] if len(parts) > 1 else '0000'
        arxiv_id = f'{left}.{right}'
        url      = f'https://arxiv.org/abs/{arxiv_id}'

        raw_clicks = int(click_store[q_key].get(float(doc_id), 0))
        raw_global = int(global_clicks.get(float(doc_id), 0))

        results.append({
            'rank':          rank,
            'id':            raw,
            'arxiv_id':      arxiv_id,
            'url':           url,
            'title':         str(doc_titles[idx]).replace('\n', ' ').strip(),
            'abstract':      str(doc_abstracts[idx])[:350].replace('\n', ' ').strip() + '...',
            'cosine_score':  round(float(cosine_scores[idx]),  4),
            'click_score':   round(float(click_scores[idx]),   4),
            'blended_score': round(float(blended_scores[idx]), 4),
            'popularity':    round(float(popularity[idx]),     4),
            'year':          int(doc_years[idx]),
            'click_count':   raw_clicks,
            'global_count':  raw_global,
        })
    return results
# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def do_search():
    query   = request.args.get('q', '').strip()
    sort_by = request.args.get('sort', 'blended')
    top_k   = int(request.args.get('k', 10))
    if not query:
        return jsonify([])
    return jsonify(search(query, top_k=top_k, sort_by=sort_by))

@app.route('/click', methods=['POST'])
def record_click():
    data   = request.get_json()
    query  = data.get('query', '')
    doc_id = data.get('doc_id', '')
    if query and doc_id:
        key = normalise_query(query)
        fid = float(doc_id)
        click_store[key][fid] += 1
        global_clicks[fid]    += 1
        with open(f'{BASE}/click_store.pkl', 'wb') as f:
            pickle.dump({k: dict(v) for k, v in click_store.items()}, f)
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
