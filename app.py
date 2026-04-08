# =============================================================================
# Method-Aware Search Engine for CS Research Papers
# Flask backend — query processing, ranking, click-through re-ranking
# =============================================================================
# Architecture overview:
#   1. At startup, load pre-built TF-IDF index and POS weight vector from disk.
#   2. /search  — preprocess query, compute cosine similarity against the
#                 POS-weighted TF-IDF matrix, blend with click history, rank.
#   3. /click   — record a user click, persist to click_store.pkl, update
#                 in-memory global popularity counts.
#   4. /        — serve the search UI (templates/index.html).
#
# Preprocessing pipeline (must match notebook indexing exactly):
#   lowercase → strip punctuation (keep hyphens) → remove digits →
#   tokenise → remove stopwords → Porter stem
# =============================================================================

from flask import Flask, render_template, request, jsonify
import pickle
import scipy.sparse as sp
import numpy as np
import re
import os
import threading
import warnings
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import InconsistentVersionWarning
from collections import defaultdict, deque

# Suppress sklearn version mismatch warning. The vectorizer pickle was built
# in Colab (sklearn 1.6.1); behaviour is confirmed correct via result inspection.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# =============================================================================
# App initialisation
# =============================================================================

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# Thread lock — prevents concurrent /click requests corrupting click_store.pkl
_click_lock = threading.Lock()


# =============================================================================
# Click deduplication
# =============================================================================
# Each click request carries a unique click_event_id generated client-side.
# We track the last MAX_CLICK_EVENT_IDS IDs in a bounded set so that retried
# or double-fired requests never increment the count more than once per action.

MAX_CLICK_EVENT_IDS = 10_000
_click_event_ids    = set()
_click_event_order  = deque()


def _is_duplicate_click(event_id: str) -> bool:
    """Return True if this event_id has already been processed."""
    return bool(event_id) and event_id in _click_event_ids


def _mark_click_processed(event_id: str) -> None:
    """Add event_id to the dedup set, evicting the oldest entry when full."""
    if not event_id or event_id in _click_event_ids:
        return
    if len(_click_event_order) >= MAX_CLICK_EVENT_IDS:
        _click_event_ids.discard(_click_event_order.popleft())
    _click_event_order.append(event_id)
    _click_event_ids.add(event_id)


# =============================================================================
# arXiv ID helpers
# =============================================================================
# Document IDs are stored as floats (e.g. 704.0033) because the arXiv dataset
# uses YYMM.NNNNN identifiers that pandas reads as floats. We reconstruct the
# canonical string form (0704.0033) for URL and display purposes.

def _float_to_arxiv_id(doc_id: float) -> str:
    """Convert a float doc_id to a zero-padded arXiv ID string.

    Example: 704.0033 → '0704.0033'
    """
    raw   = str(doc_id)
    parts = raw.split('.')
    left  = parts[0].zfill(4)
    right = parts[1] if len(parts) > 1 else '0000'
    return f'{left}.{right}'


def _arxiv_url(doc_id: float) -> str:
    """Return the full arXiv abstract URL for a document."""
    return f'https://arxiv.org/abs/{_float_to_arxiv_id(doc_id)}'


# =============================================================================
# Index loading
# =============================================================================

def _load_index() -> tuple:
    """Load all pre-built search artifacts from disk at startup.

    Returns:
        vectorizer        — fitted TfidfVectorizer (for query transformation)
        pos_weight_diag   — sparse diagonal matrix of POS multipliers
        doc_ids           — list of float arXiv IDs aligned to matrix rows
        doc_titles        — list of paper titles
        doc_abstracts     — list of paper abstracts
        tfidf_matrix      — baseline TF-IDF matrix (no POS weighting)
        tfidf_pos_matrix  — POS-weighted TF-IDF matrix (used for ranking)
        click_store       — dict[query_key → dict[doc_id → click_count]]
        doc_years         — list of int publication years
        doc_categories    — list of category strings aligned to rows (optional)
        global_clicks     — dict[doc_id → total clicks across all queries]

    Note:
        When doc_authors.pkl and doc_author_tokens.pkl are present in the repo
        root (generated by the notebook BL-012 cell), author-keyword boosting is
        applied automatically. When absent, ranking falls back to current behaviour.
    """
    # --- Core index artifacts ---
    with open(f'{BASE}/vectorizer.pkl',        'rb') as f: vectorizer        = pickle.load(f)
    with open(f'{BASE}/pos_weight_vector.pkl', 'rb') as f: pos_weight_vector = pickle.load(f)
    with open(f'{BASE}/doc_ids.pkl',           'rb') as f: doc_ids           = pickle.load(f)
    with open(f'{BASE}/doc_titles.pkl',        'rb') as f: doc_titles        = pickle.load(f)
    with open(f'{BASE}/doc_abstracts.pkl',     'rb') as f: doc_abstracts     = pickle.load(f)

    tfidf_matrix     = sp.load_npz(f'{BASE}/tfidf_base_matrix.npz')
    tfidf_pos_matrix = sp.load_npz(f'{BASE}/tfidf_pos_matrix.npz')

    # --- Click store ---
    # Normalise all doc_id keys to float and counts to int on load so types
    # are consistent throughout the session regardless of how they were saved.
    try:
        with open(f'{BASE}/click_store.pkl', 'rb') as f:
            raw_store = pickle.load(f)
        click_store: defaultdict = defaultdict(lambda: defaultdict(int))
        for query_key, doc_clicks in raw_store.items():
            for doc_id, count in doc_clicks.items():
                click_store[query_key][float(doc_id)] += int(count)
    except Exception:
        click_store = defaultdict(lambda: defaultdict(int))

    # --- Publication years and categories ---
    # Prefer explicit serialized metadata when available; fall back to
    # deterministic derivation (for years) or disabled filtering (categories).
    try:
        with open(f'{BASE}/doc_years.pkl', 'rb') as f:
            doc_years = [int(y) for y in pickle.load(f)]
        if len(doc_years) != len(doc_ids):
            print('WARNING: doc_years length mismatch — deriving years from IDs')
            doc_years = []
    except FileNotFoundError:
        doc_years = []

    if not doc_years:
        # arXiv IDs encode the submission year as the first two digits of the
        # integer part (YYMM format). Values < 50 are treated as 20YY, >= 50
        # as 19YY. Malformed IDs default to 0.
        for doc_id in doc_ids:
            try:
                prefix = str(doc_id).split('.')[0].zfill(4)
                yy     = int(prefix[:2])
                doc_years.append(2000 + yy if yy < 50 else 1900 + yy)
            except (ValueError, IndexError):
                doc_years.append(0)

    try:
        with open(f'{BASE}/doc_categories.pkl', 'rb') as f:
            doc_categories = [str(v) for v in pickle.load(f)]
        if len(doc_categories) != len(doc_ids):
            print('WARNING: doc_categories length mismatch — category filter disabled')
            doc_categories = None
    except FileNotFoundError:
        doc_categories = None

    # --- Global click popularity ---
    # Aggregate click counts across all queries so we can rank by overall
    # popularity independently of the current query.
    global_clicks: defaultdict = defaultdict(int)
    for doc_clicks in click_store.values():
        for doc_id, count in doc_clicks.items():
            global_clicks[float(doc_id)] += int(count)

    pos_weight_diag = sp.diags(pos_weight_vector)

    # --- Author metadata (optional — BL-012) ---
    # Loaded with fallback so the server starts even before the user runs
    # the updated notebook and supplies the two new artifact files.
    try:
        with open(f'{BASE}/doc_authors.pkl', 'rb') as f:
            doc_authors = pickle.load(f)
        with open(f'{BASE}/doc_author_tokens.pkl', 'rb') as f:
            doc_author_tokens = pickle.load(f)
        if len(doc_authors) != len(doc_ids) or len(doc_author_tokens) != len(doc_ids):
            print('WARNING: author artifact length mismatch — author boost disabled')
            doc_authors, doc_author_tokens = None, None
        else:
            print(f'Author artifacts loaded — boost enabled for {len(doc_authors)} docs')
    except FileNotFoundError:
        doc_authors, doc_author_tokens = None, None

    return (vectorizer, pos_weight_diag, doc_ids, doc_titles,
            doc_abstracts, tfidf_matrix, tfidf_pos_matrix,
            click_store, doc_years, doc_categories, global_clicks,
            doc_authors, doc_author_tokens)


# Load once at startup — all search functions reference these module-level names
print("Loading index...")
(vectorizer, pos_weight_diag, doc_ids, doc_titles,
 doc_abstracts, tfidf_matrix, tfidf_pos_matrix,
 click_store, doc_years, doc_categories, global_clicks,
 doc_authors, doc_author_tokens) = _load_index()

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
feature_names = vectorizer.get_feature_names_out()  # vocabulary aligned to matrix columns
print(f"Index loaded — {len(doc_ids)} documents ready")

# Cache conservative WordNet expansions by token to avoid repeated lookups.
_WORDNET_CACHE: dict[str, list[str]] = {}


# =============================================================================
# NLP / query preprocessing
# =============================================================================
# This pipeline MUST match the preprocessing applied to documents during
# notebook indexing. Any divergence will cause incorrect cosine scores.

def _expand_query_with_wordnet(tokens: list[str], stemmed_tokens: list[str]) -> list[str]:
    """Expand query tokens conservatively using WordNet synonyms.

    Rules:
        - Original stemmed tokens are always retained.
        - At most the first two synsets and first two lemmas are considered.
        - Multi-word lemmas are split, stemmed, and appended uniquely.
        - If WordNet data is unavailable, expansion is silently skipped.
    """
    expanded = list(stemmed_tokens)
    seen = set(expanded)

    for tok in tokens:
        if tok in _WORDNET_CACHE:
            extra_stems = _WORDNET_CACHE[tok]
        else:
            try:
                synsets = wordnet.synsets(tok)[:2]
            except LookupError:
                return expanded

            extra_stems = []
            for syn in synsets:
                for lemma in syn.lemmas()[:2]:
                    lemma_name = lemma.name().lower().replace('_', ' ').strip()
                    if not lemma_name:
                        continue
                    for part in lemma_name.split():
                        stemmed = stemmer.stem(part)
                        if stemmed and stemmed not in extra_stems:
                            extra_stems.append(stemmed)
            _WORDNET_CACHE[tok] = extra_stems

        for candidate in extra_stems:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)

    return expanded


def _preprocess(text: str, expand_terms: bool = False) -> str:
    """Apply the full indexing preprocessing pipeline to a text string.

    Steps:
        1. Lowercase
        2. Remove punctuation (hyphens preserved for compound terms)
        3. Remove digits
        4. Tokenise (NLTK word_tokenize)
        5. Remove stopwords
        6. Porter stemming
    """
    text   = text.lower()
    text   = re.sub(r'[^\w\s\-]', '', text)   # strip punctuation, keep hyphens
    text   = re.sub(r'\d+', '', text)          # remove digits
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.strip()]
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    if expand_terms:
        stemmed_tokens = _expand_query_with_wordnet(tokens, stemmed_tokens)

    return ' '.join(stemmed_tokens)


def _normalise_categories(raw_categories: list[str] | None) -> list[str]:
    """Normalise category filter values to lowercase tokens for contains-match."""
    if not raw_categories:
        return []

    out: list[str] = []
    for value in raw_categories:
        for piece in str(value).split(','):
            token = piece.strip().lower()
            if token and token not in out:
                out.append(token)
    return out


def _normalise_query(query: str) -> str:
    """Return a canonical, order-independent click store key for a query.

    Tokens are sorted alphabetically so 'deep learning' and 'learning deep'
    map to the same key, preventing click history fragmentation.
    """
    return ' '.join(sorted(_preprocess(query).split()))


# =============================================================================
# Scoring helpers
# =============================================================================

def _get_click_score(query: str, doc_id: float) -> float:
    """Return normalised query-specific click score in [0, 1].

    Score = clicks(doc, query) / max_clicks_for_query.
    The most-clicked document for a query scores 1.0; unseen docs score 0.0.
    """
    key    = _normalise_query(query)
    clicks = click_store[key]
    if not clicks:
        return 0.0
    count = clicks.get(float(doc_id), 0)
    return count / max(clicks.values())


def _get_global_popularity(doc_id: float) -> float:
    """Return normalised global click popularity in [0, 1] across all queries."""
    if not global_clicks:
        return 0.0
    return global_clicks.get(float(doc_id), 0) / max(global_clicks.values())


# =============================================================================
# Explainability helper
# =============================================================================

def _top_terms_for(doc_idx: int, nonzero_cols: np.ndarray, query_vec_array: np.ndarray) -> list[str]:
    """Return the top _TOP_TERMS_K vocabulary terms that contributed most to
    the cosine score between the query and document at *doc_idx*.

    Method: for each non-zero query column, multiply the query weight by the
    document's TF-IDF weight for that term. The top-scoring terms are the
    ones that drove the match.
    """
    if len(nonzero_cols) == 0:
        return []

    doc_row = np.asarray(tfidf_pos_matrix[doc_idx, :][:, nonzero_cols].todense()).flatten()
    contributions = query_vec_array[nonzero_cols] * doc_row

    max_contrib = float(contributions.max())
    if max_contrib <= 0:
        return []

    top_local = contributions.argsort()[::-1][:_TOP_TERMS_K]
    return [str(feature_names[nonzero_cols[i]]) for i in top_local if contributions[i] > 0]


# =============================================================================
# Search
# =============================================================================

# Required fields in every result dict — validated before returning to client
_REQUIRED_FIELDS = {
    'rank', 'id', 'arxiv_id', 'url', 'title', 'abstract',
    'cosine_score', 'click_score', 'blended_score',
    'popularity', 'year', 'click_count', 'global_count', 'top_terms',
}

# Number of top matching terms to return per result for explainability display
_TOP_TERMS_K = 5

# Blending weight: final_score = ALPHA * cosine + (1 - ALPHA) * click_score
# 0.7 keeps retrieval quality dominant while letting click history influence
# ranking. Lower values give more weight to popularity (see alpha analysis
# in notebook).
ALPHA = 0.7

# Minimum cosine score a document must have to appear in year/popularity sorts.
# Prevents irrelevant documents being surfaced purely because they are old/popular.
_RELEVANCE_THRESHOLD = 0.05


def search(
    query: str,
    top_k: int = 10,
    sort_by: str = 'blended',
    categories: list[str] | None = None,
    expand_terms: bool = False,
) -> list:
    """Run a search query and return a ranked list of results.

    Args:
        query:   Raw user query string (preprocessed internally).
        top_k:   Maximum number of results to return.
        sort_by: Ranking strategy. Options:
                   'blended'    cosine (0.7) + query click score (0.3)  [default]
                   'relevance'  pure cosine similarity — no click influence
                   'trending'   cosine (0.6) + global popularity (0.4)
                   'popularity' global click count (relevant docs only)
                   'newest'     publication year descending (relevant docs only)
                   'oldest'     publication year ascending (relevant docs only)

    BL-012:
        When doc_author_tokens is loaded, query stem tokens that match any
        author name token receive a 1.5x per-document multiplier on cosine_scores
        before blending. Falls back to unmodified cosine when artifacts are absent.

    Returns:
        List of result dicts each containing all _REQUIRED_FIELDS keys.

    Raises:
        ValueError: If a result dict is missing any required field.
    """
    # Step 1 — preprocess and vectorise the query
    processed     = _preprocess(query, expand_terms=expand_terms)
    query_vec     = vectorizer.transform([processed])
    query_vec_pos = query_vec.dot(pos_weight_diag)   # apply POS multipliers

    # Step 2 — compute score arrays for all documents
    cosine_scores  = cosine_similarity(query_vec_pos, tfidf_pos_matrix).flatten()

    # Author-keyword boost (BL-012): if author artifacts are loaded, apply a
    # 1.5x multiplier to cosine_scores for documents whose author tokens
    # overlap with the preprocessed query tokens.
    if doc_author_tokens is not None:
        query_stems = set(processed.split())
        author_boost = np.array(
            [1.5 if query_stems & doc_author_tokens[i] else 1.0
             for i in range(len(doc_ids))],
            dtype=float,
        )
        cosine_scores = cosine_scores * author_boost

    click_scores   = np.array([_get_click_score(query, doc_ids[i])    for i in range(len(doc_ids))])
    blended_scores = ALPHA * cosine_scores + (1 - ALPHA) * click_scores
    year_scores    = np.array(doc_years, dtype=float)
    popularity     = np.array([_get_global_popularity(doc_ids[i])     for i in range(len(doc_ids))])

    # Step 3 — select the ranking array based on sort_by
    if sort_by == 'relevance':
        scores = cosine_scores
    elif sort_by == 'trending':
        scores = 0.6 * cosine_scores + 0.4 * popularity
    elif sort_by == 'popularity':
        scores = np.where(cosine_scores > _RELEVANCE_THRESHOLD, popularity, 0)
    elif sort_by == 'newest':
        scores = np.where(cosine_scores > _RELEVANCE_THRESHOLD, year_scores, 0)
    elif sort_by == 'oldest':
        max_year = float(max(doc_years)) if doc_years else 2026.0
        scores   = np.where(cosine_scores > _RELEVANCE_THRESHOLD, max_year - year_scores, 0)
    else:  # 'blended' (default)
        scores = blended_scores

    # Step 4 — rank and optionally apply category filtering
    ranked_indices = scores.argsort()[::-1]
    category_tokens = _normalise_categories(categories)

    if category_tokens and doc_categories is not None:
        top_indices = []
        for idx in ranked_indices:
            category_value = str(doc_categories[idx]).lower()
            if any(token in category_value for token in category_tokens):
                top_indices.append(int(idx))
            if len(top_indices) >= top_k:
                break
    else:
        top_indices = ranked_indices[:top_k]

    q_key       = _normalise_query(query)

    # Pre-compute query contribution vector once — element-wise product of the
    # POS-weighted query vector and each document row gives the per-term score.
    # We extract the non-zero query column indices so the inner loop is fast.
    query_vec_array = np.asarray(query_vec_pos.todense()).flatten()
    nonzero_cols = np.where(query_vec_array > 0)[0]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id   = doc_ids[idx]
        arxiv_id = _float_to_arxiv_id(doc_id)

        row = {
            'rank':          rank,
            'id':            str(doc_id),
            'arxiv_id':      arxiv_id,
            'url':           _arxiv_url(doc_id),
            'title':         str(doc_titles[idx]).replace('\n', ' ').strip(),
            'abstract':      str(doc_abstracts[idx])[:350].replace('\n', ' ').strip() + '...',
            'cosine_score':  round(float(cosine_scores[idx]),  4),
            'click_score':   round(float(click_scores[idx]),   4),
            'blended_score': round(float(blended_scores[idx]), 4),
            'popularity':    round(float(popularity[idx]),     4),
            'year':          int(doc_years[idx]),
            'click_count':   int(click_store[q_key].get(float(doc_id), 0)),
            'global_count':  int(global_clicks.get(float(doc_id), 0)),
            'top_terms':     _top_terms_for(idx, nonzero_cols, query_vec_array),
        }

        missing = _REQUIRED_FIELDS.difference(row.keys())
        if missing:
            raise ValueError(f'Missing search response fields: {sorted(missing)}')

        results.append(row)

    return results


# =============================================================================
# Flask routes
# =============================================================================

@app.route('/')
def index():
    """Serve the search UI."""
    return render_template('index.html')


@app.route('/search')
def do_search():
    """Handle a search request and return ranked results as JSON.

    Query parameters:
        q     (str)  — search query (required)
        sort  (str)  — sort strategy, default 'blended'
        k     (int)  — number of results, default 10

    Returns:
        200 JSON array of result objects on success.
        200 empty array if query is blank.
        500 JSON {'error': ...} if result validation fails.
    """
    query   = request.args.get('q', '').strip()
    sort_by = request.args.get('sort', 'blended')
    top_k   = int(request.args.get('k', 10))
    expand_terms = request.args.get('expand', '0').strip().lower() in ('1', 'true', 'yes', 'on')

    categories = request.args.getlist('cat')
    if not categories:
        single_cat = request.args.get('cat', '').strip()
        if single_cat:
            categories = [single_cat]

    if not query:
        return jsonify([])

    try:
        return jsonify(
            search(
                query,
                top_k=top_k,
                sort_by=sort_by,
                categories=categories,
                expand_terms=expand_terms,
            )
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/click', methods=['POST'])
def record_click():
    """Record a user click on a search result.

    Expected JSON body:
        query          (str) — the query that produced the result
        doc_id         (str) — the document's float ID as a string
        click_event_id (str) — client-generated UUID for deduplication

    Deduplication guarantees exactly one increment per user action even if
    the title-click and open-button both fire, or if the client retries.

    In-memory click_store and global_clicks are updated immediately.
    click_store.pkl is written to disk under a thread lock.

    Returns:
        JSON {'status': 'ok' | 'deduplicated' | 'invalid_doc_id'}
    """
    data           = request.get_json(silent=True) or {}
    query          = str(data.get('query',          '')).strip()
    doc_id_str     = str(data.get('doc_id',         '')).strip()
    click_event_id = str(data.get('click_event_id', '')).strip()

    if _is_duplicate_click(click_event_id):
        return jsonify({'status': 'deduplicated'})

    if query and doc_id_str:
        try:
            fid = float(doc_id_str)
        except ValueError:
            return jsonify({'status': 'invalid_doc_id'}), 400

        key = _normalise_query(query)

        # Update in-memory stores — int counts throughout
        click_store[key][fid] += 1
        global_clicks[fid]    += 1

        # Persist to disk under thread lock
        with _click_lock:
            with open(f'{BASE}/click_store.pkl', 'wb') as f:
                pickle.dump({k: dict(v) for k, v in click_store.items()}, f)

        _mark_click_processed(click_event_id)

    return jsonify({'status': 'ok'})


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
