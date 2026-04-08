"""Minimal local Flask smoke test for BL-019/BL-012 checks.

Checks:
1) GET / returns 200 and HTML.
2) GET /search returns 200 and valid result schema.
3) POST /click persists and updates click_store.pkl mtime.
4) Duplicate click_event_id returns deduplicated status.
5) Invalid doc_id returns 400 invalid_doc_id.
6) Query normalisation remains order-insensitive (parity safeguard).
7) Newest/oldest/popularity sorts remain relevance-threshold gated.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

from app import (
    BASE,
    _RELEVANCE_THRESHOLD,
    _REQUIRED_FIELDS,
    _normalise_query,
    app,
    doc_categories,
    doc_ids,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        _fail(msg)


def _expect_status(resp: Any, expected: int, name: str) -> None:
    _assert(
        resp.status_code == expected,
        f"{name}: expected HTTP {expected}, got {resp.status_code}",
    )


def _wait_for_mtime_change(path: str, before: float, timeout_s: float = 2.5) -> bool:
    # Handle coarse filesystem timestamp resolution with a short bounded poll.
    end = time.time() + timeout_s
    while time.time() < end:
        if os.path.getmtime(path) > before:
            return True
        time.sleep(0.1)
    return os.path.getmtime(path) > before


def run() -> int:
    failures: list[str] = []

    try:
        click_store_path = os.path.join(BASE, "click_store.pkl")
        _assert(os.path.exists(click_store_path), "click_store.pkl does not exist")

        # BL-012 soft check: if author artifacts exist, verify length alignment.
        # Skipped (not a failure) when files are not yet present — Flask runtime
        # falls back to no-boost behaviour in that case.
        for art in ("doc_authors.pkl", "doc_author_tokens.pkl"):
            art_path = os.path.join(BASE, art)
            if os.path.exists(art_path):
                import pickle as _pkl
                with open(art_path, "rb") as _f:
                    _art = _pkl.load(_f)
                _assert(
                    len(_art) == len(doc_ids),
                    f"{art} length {len(_art)} does not match doc_ids length {len(doc_ids)}",
                )
            else:
                print(f"  [skip] {art} not present — author boost inactive until runbook is re-run")

        with app.test_client() as client:
            doc_id_to_index = {str(v): i for i, v in enumerate(doc_ids)}

            # BL-012 parity safeguard: normalisation should be order-insensitive.
            _assert(
                _normalise_query("deep learning") == _normalise_query("learning deep"),
                "Query normalisation parity failed for token order variation",
            )

            # 1) GET /
            root_resp = client.get("/")
            _expect_status(root_resp, 200, "GET /")
            body = root_resp.get_data(as_text=True)
            _assert("<html" in body.lower(), "GET / did not return HTML content")

            # 2) GET /search
            query = "deep learning image classification"
            search_resp = client.get(
                "/search",
                query_string={"q": query, "sort": "blended", "k": 5},
            )
            _expect_status(search_resp, 200, "GET /search")
            results = search_resp.get_json()
            _assert(isinstance(results, list), "GET /search did not return a JSON list")
            _assert(len(results) > 0, "GET /search returned no results for smoke query")
            first = results[0]
            _assert(isinstance(first, dict), "Search result item is not an object")
            missing = sorted(_REQUIRED_FIELDS.difference(first.keys()))
            _assert(not missing, f"Search result missing required fields: {missing}")

            doc_id = str(first["id"])

            # 3) POST /click valid + mtime change
            before_mtime = os.path.getmtime(click_store_path)
            event_id = str(uuid.uuid4())
            click_payload = {
                "query": query,
                "doc_id": doc_id,
                "click_event_id": event_id,
            }
            click_resp = client.post("/click", json=click_payload)
            _expect_status(click_resp, 200, "POST /click valid")
            click_json = click_resp.get_json() or {}
            _assert(click_json.get("status") == "ok", f"POST /click valid status was {click_json}")

            changed = _wait_for_mtime_change(click_store_path, before_mtime)
            _assert(changed, "click_store.pkl mtime did not change after valid click")

            # 4) POST /click duplicate event id
            dup_resp = client.post("/click", json=click_payload)
            _expect_status(dup_resp, 200, "POST /click duplicate")
            dup_json = dup_resp.get_json() or {}
            _assert(
                dup_json.get("status") == "deduplicated",
                f"POST /click duplicate status was {dup_json}",
            )

            # 5) POST /click invalid doc_id
            bad_payload = {
                "query": query,
                "doc_id": "not_a_float",
                "click_event_id": str(uuid.uuid4()),
            }
            bad_resp = client.post("/click", json=bad_payload)
            _expect_status(bad_resp, 400, "POST /click invalid doc_id")
            bad_json = bad_resp.get_json() or {}
            _assert(
                bad_json.get("status") == "invalid_doc_id",
                f"POST /click invalid doc_id status was {bad_json}",
            )

            # 6) Sort modes should still enforce relevance threshold gating.
            for sort_mode in ("newest", "oldest", "popularity"):
                sort_resp = client.get(
                    "/search",
                    query_string={"q": query, "sort": sort_mode, "k": 5},
                )
                _expect_status(sort_resp, 200, f"GET /search sort={sort_mode}")
                sort_results = sort_resp.get_json() or []
                _assert(isinstance(sort_results, list), f"sort={sort_mode} did not return list")
                _assert(len(sort_results) > 0, f"sort={sort_mode} returned no results")

                top_cosine = float(sort_results[0]["cosine_score"])
                _assert(
                    top_cosine > _RELEVANCE_THRESHOLD,
                    (
                        f"sort={sort_mode} top result cosine_score={top_cosine} "
                        f"did not clear threshold={_RELEVANCE_THRESHOLD}"
                    ),
                )

            # 7) Expansion toggle should keep endpoint healthy in both modes.
            expand_off = client.get(
                "/search",
                query_string={"q": query, "sort": "blended", "k": 5, "expand": "0"},
            )
            _expect_status(expand_off, 200, "GET /search expand=0")
            _assert(isinstance(expand_off.get_json(), list), "expand=0 response is not list")

            expand_on = client.get(
                "/search",
                query_string={"q": query, "sort": "blended", "k": 5, "expand": "1"},
            )
            _expect_status(expand_on, 200, "GET /search expand=1")
            _assert(isinstance(expand_on.get_json(), list), "expand=1 response is not list")

            # 8) Category filtering checks (when category artifact is present).
            if doc_categories is not None:
                category_tokens: list[str] = []
                for raw in doc_categories:
                    for token in str(raw).split("|"):
                        token = token.strip()
                        if token and token not in category_tokens:
                            category_tokens.append(token)
                    if len(category_tokens) >= 2:
                        break

                if category_tokens:
                    cat1 = category_tokens[0]
                    cat_resp = client.get(
                        "/search",
                        query_string={"q": query, "sort": "blended", "k": 5, "cat": cat1},
                    )
                    _expect_status(cat_resp, 200, "GET /search single cat")
                    for row in cat_resp.get_json() or []:
                        idx = doc_id_to_index.get(str(row["id"]))
                        _assert(idx is not None, "single-cat result doc id missing from doc_ids")
                        doc_cat = str(doc_categories[idx]).lower()
                        _assert(cat1.lower() in doc_cat, f"single-cat filter mismatch for {row['id']}")

                if len(category_tokens) >= 2:
                    cat1, cat2 = category_tokens[0], category_tokens[1]
                    multi_resp = client.get(
                        "/search",
                        query_string=[
                            ("q", query),
                            ("sort", "blended"),
                            ("k", 5),
                            ("cat", cat1),
                            ("cat", cat2),
                        ],
                    )
                    _expect_status(multi_resp, 200, "GET /search multi cat")
                    for row in multi_resp.get_json() or []:
                        idx = doc_id_to_index.get(str(row["id"]))
                        _assert(idx is not None, "multi-cat result doc id missing from doc_ids")
                        doc_cat = str(doc_categories[idx]).lower()
                        _assert(
                            (cat1.lower() in doc_cat) or (cat2.lower() in doc_cat),
                            f"multi-cat filter mismatch for {row['id']}",
                        )
            else:
                print("  [skip] doc_categories.pkl not present — category filter assertions skipped")

    except Exception as exc:  # noqa: BLE001 - smoke test should report any failure
        failures.append(str(exc))

    if failures:
        print("SMOKE TEST: FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("SMOKE TEST: PASS")
    print("- GET / returned 200 HTML")
    print("- GET /search returned 200 with required schema")
    print("- POST /click valid returned ok and updated click_store.pkl mtime")
    print("- POST /click duplicate returned deduplicated")
    print("- POST /click invalid doc_id returned 400 invalid_doc_id")
    print("- Query normalisation remained order-insensitive")
    print("- Newest/oldest/popularity respected relevance threshold gating")
    print("- Expansion toggle handled /search requests in both modes")
    print("- Category filter behavior verified (or skipped if category artifact absent)")
    print("- Author artifact length alignment verified (or skipped if PKLs absent)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
