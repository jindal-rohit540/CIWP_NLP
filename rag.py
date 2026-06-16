"""RAG layer for CIWP Intelligence.

The CIWP extract is a *denormalized hierarchy*, not one-row-per-plan:

    School (Plan) -> Priority (1..4) -> [one SCP/RC/ToA chain] + many goal/metric rows

The SCP, Root Cause and Theory of Action are identical across every row of a
priority (verified: 0/1249 groups vary), while goals / metrics / student groups
vary row-by-row. So the natural retrieval unit is the **(School x Priority)
logic chain** — 1,249 self-contained chunks — with the per-row goals rolled up.

Retrieval is filter-first: sidebar filters become hard *metadata* constraints on
the vector store, and semantic similarity ranks within that scope. This replaces
the old "serialize every filtered row, cap at 150, even-sample" approach.

Embeddings: OpenAI text-embedding-3-small. Store: persistent ChromaDB (./chroma).
"""

import os
import re
import html
import hashlib

import pandas as pd
import chromadb
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "ciwp_chains"
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")
EMBED_BATCH = 256  # rows per embedding API call

# Source column names (mirrors COL_MAP in app.py).
C_SCHOOL = "Plan Name"
C_PRIORITY = "Priority Name"
C_FOUNDATION = "Foundation"
C_NETWORK = "Network"
C_SCP = "Student-Centered Problem"
C_RC = "Root Cause (adult-facing)"
C_IF = "ToA: If We"
C_THEN = "ToA: Then We See"
C_LEADS = "ToA: Which Leads To"
C_G1, C_G1T = "Year 1 Practice Goal", "Year 1 Practice Goal Target"
C_G2, C_G3 = "Year 2 Practice Goal", "Year 3 Practice Goal"
C_PERF, C_METRIC, C_SG = "Performance Goal", "Metric", "Student Group"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean(v) -> str:
    return html.unescape(str(v)).strip() if pd.notna(v) else ""


def _school_name(plan_name: str) -> str:
    return re.sub(r"\s+CIWP Cycle.*$", "", str(plan_name), flags=re.IGNORECASE).strip()


def _priority_number(priority_name: str):
    m = re.match(r"\s*Priority\s*(\d+)", str(priority_name))
    return int(m.group(1)) if m else 0


def _api_key() -> str:
    # Match app.py: Streamlit secrets first (cloud), then .env (local).
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("OPENAI_API_KEY", "")


def _chunk_id(school: str, priority: str) -> str:
    """Stable ID so re-indexing / uploads upsert instead of duplicating."""
    return hashlib.sha1(f"{school}||{priority}".encode("utf-8")).hexdigest()


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_plans(df: pd.DataFrame):
    """Collapse the denormalized rows into one chunk per (School x Priority).

    Returns (ids, documents, metadatas) ready for a Chroma collection.
    """
    ids, docs, metas = [], [], []

    for (plan, priority), grp in df.groupby([C_SCHOOL, C_PRIORITY], sort=False):
        first = grp.iloc[0]
        school = _school_name(plan)
        foundation = _clean(first.get(C_FOUNDATION, "")) or re.sub(
            r"^\s*Priority\s*\d+\s*:\s*", "", str(priority)
        ).strip()
        network = _clean(first.get(C_NETWORK, ""))

        # Roll up the goal/metric/student-group rows that vary within the group.
        goal_lines, perf_lines, groups = [], [], set()
        for _, r in grp.iterrows():
            g1, g1t = _clean(r.get(C_G1)), _clean(r.get(C_G1T))
            g2, g3 = _clean(r.get(C_G2)), _clean(r.get(C_G3))
            if g1:
                goal_lines.append(f"Y1: {g1}" + (f" (target: {g1t})" if g1t else ""))
            if g2:
                goal_lines.append(f"Y2: {g2}")
            if g3:
                goal_lines.append(f"Y3: {g3}")
            perf, metric = _clean(r.get(C_PERF)), _clean(r.get(C_METRIC))
            if perf or metric:
                perf_lines.append(" / ".join(p for p in [perf, metric] if p))
            sg = _clean(r.get(C_SG))
            if sg:
                groups.add(sg)

        # The embedded document: the full logic chain, stated once.
        doc = (
            f"SCHOOL: {school}\n"
            f"NETWORK: {network}\n"
            f"PRIORITY: {priority}\n"
            f"STUDENT GROUPS: {', '.join(sorted(groups)) if groups else 'Not specified'}\n"
            f"STUDENT-CENTERED PROBLEM: {_clean(first.get(C_SCP))}\n"
            f"ROOT CAUSE: {_clean(first.get(C_RC))}\n"
            f"THEORY OF ACTION — IF WE: {_clean(first.get(C_IF))}\n"
            f"THEN WE SEE: {_clean(first.get(C_THEN))}\n"
            f"WHICH LEADS TO: {_clean(first.get(C_LEADS))}\n"
            f"PRACTICE GOALS:\n  " + ("\n  ".join(goal_lines) if goal_lines else "None stated") + "\n"
            f"PERFORMANCE GOALS:\n  " + ("\n  ".join(perf_lines) if perf_lines else "None stated")
        )

        ids.append(_chunk_id(school, str(priority)))
        docs.append(doc)
        metas.append({
            "school": school,
            "network": network,
            "foundation": foundation,
            "priority_name": str(priority),
            "priority_number": _priority_number(priority),
            "n_rows": int(len(grp)),
            "student_groups": ", ".join(sorted(groups)),
        })

    return ids, docs, metas


# ── Embeddings ────────────────────────────────────────────────────────────────
def _embed(texts, client: OpenAI):
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


# ── Index lifecycle ───────────────────────────────────────────────────────────
def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    return _client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def index_status() -> dict:
    """Cheap check used by the UI to decide whether to prompt a build."""
    try:
        col = get_collection()
        return {"ready": True, "count": col.count()}
    except Exception as e:
        return {"ready": False, "count": 0, "error": str(e)}


def build_index(df: pd.DataFrame, upsert: bool = True, progress=None):
    """Chunk df, embed, and upsert into Chroma. Returns the number of chunks.

    `upsert=True` adds/updates without wiping the store (used for uploads).
    `progress` is an optional callback(done, total) for a Streamlit progress bar.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured (Streamlit secrets or .env).")

    ids, docs, metas = chunk_plans(df)
    if not ids:
        return 0

    client = OpenAI(api_key=key)
    col = get_collection()
    if not upsert:
        # Fresh rebuild: drop and recreate the collection.
        _client().delete_collection(COLLECTION_NAME)
        col = get_collection()

    total = len(ids)
    for i in range(0, total, EMBED_BATCH):
        sl = slice(i, i + EMBED_BATCH)
        embs = _embed(docs[sl], client)
        col.upsert(ids=ids[sl], documents=docs[sl], metadatas=metas[sl], embeddings=embs)
        if progress:
            progress(min(i + EMBED_BATCH, total), total)
    return total


# ── Retrieval ─────────────────────────────────────────────────────────────────
# Hybrid = dense (Chroma embeddings, semantic) + sparse (BM25, lexical), fused
# with Reciprocal Rank Fusion, then a light re-anchor rerank that boosts chunks
# literally containing the query's salient terms (catches acronyms: MTSS, IAR…).

RRF_K = 60          # RRF damping constant
DENSE_WEIGHT = 1.0  # relative weight of the dense ranker in fusion
SPARSE_WEIGHT = 1.0  # relative weight of the BM25 ranker in fusion
ANCHOR_BOOST = 0.15  # rerank bump per query term present in a chunk


def _where(network: str = None, foundation: str = None, priority_number: int = None):
    """Build a Chroma metadata filter from the active sidebar filters."""
    clauses = []
    if network and network != "All Networks":
        clauses.append({"network": network})
    if foundation and foundation != "All Priorities":
        clauses.append({"foundation": foundation})
    if priority_number:
        clauses.append({"priority_number": priority_number})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


# Cached BM25 corpus: built once from the whole collection, sliced per-query by
# the in-scope chunk ids so the metadata filter still applies to sparse search.
_BM25_CACHE = {"sig": None, "ids": None, "metas": None, "docs": None, "bm25": None}


def _ensure_bm25():
    from rank_bm25 import BM25Okapi
    col = get_collection()
    sig = col.count()
    if _BM25_CACHE["sig"] == sig and _BM25_CACHE["bm25"] is not None:
        return _BM25_CACHE
    got = col.get(include=["documents", "metadatas"])
    ids = got.get("ids", [])
    docs = got.get("documents", [])
    metas = got.get("metadatas", [])
    _BM25_CACHE.update(
        sig=sig, ids=ids, docs=docs, metas=metas,
        bm25=BM25Okapi([_tokenize(d) for d in docs]) if docs else None,
    )
    return _BM25_CACHE


def _meta_match(meta, network, foundation, priority_number) -> bool:
    if network and network != "All Networks" and meta.get("network") != network:
        return False
    if foundation and foundation != "All Priorities" and meta.get("foundation") != foundation:
        return False
    if priority_number and meta.get("priority_number") != priority_number:
        return False
    return True


def _dense_ranking(query, client, where, k):
    """Return ordered list of (chunk_id, payload) from the vector store."""
    col = get_collection()
    q_emb = _embed([query], client)[0]
    res = col.query(
        query_embeddings=[q_emb], n_results=k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [
        (cid, {"document": d, "metadata": m, "distance": dist})
        for cid, d, m, dist in zip(ids, docs, metas, dists)
    ]


def _sparse_ranking(query, where_args, k):
    """BM25 ranking restricted to chunks matching the metadata filter."""
    cache = _ensure_bm25()
    if not cache["bm25"]:
        return []
    network, foundation, priority_number = where_args
    scores = cache["bm25"].get_scores(_tokenize(query))
    scored = [
        (cache["ids"][i], scores[i], cache["docs"][i], cache["metas"][i])
        for i in range(len(cache["ids"]))
        if _meta_match(cache["metas"][i], network, foundation, priority_number)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        (cid, {"document": d, "metadata": m, "distance": None})
        for cid, _s, d, m in scored[:k]
    ]


def _rrf_fuse(dense, sparse):
    """Reciprocal Rank Fusion of two ranked (id, payload) lists.

    Also records provenance per chunk so the UI can explain *how* it was found:
    which ranker(s) surfaced it and at what rank.
    """
    scores, payloads, prov = {}, {}, {}
    for rank, (cid, p) in enumerate(dense):
        scores[cid] = scores.get(cid, 0.0) + DENSE_WEIGHT / (RRF_K + rank + 1)
        payloads[cid] = p
        prov.setdefault(cid, {})["dense_rank"] = rank + 1
    for rank, (cid, p) in enumerate(sparse):
        scores[cid] = scores.get(cid, 0.0) + SPARSE_WEIGHT / (RRF_K + rank + 1)
        payloads.setdefault(cid, p)
        prov.setdefault(cid, {})["sparse_rank"] = rank + 1
    return scores, payloads, prov


def _reanchor(query, scores, payloads, prov):
    """Light rerank: boost chunks whose text literally contains query terms.

    Records the matched terms per chunk for UI transparency.
    """
    terms = set(_tokenize(query))
    # Drop very common, non-discriminative words so the boost rewards real signal.
    stop = {"the", "a", "an", "of", "in", "and", "or", "to", "for", "are", "is",
            "which", "what", "schools", "school", "their", "any", "how", "across"}
    terms -= stop
    if not terms:
        return scores
    for cid, p in payloads.items():
        toks = set(_tokenize(p["document"]))
        matched = sorted(terms & toks)
        if matched:
            scores[cid] += ANCHOR_BOOST * len(matched)
            prov.setdefault(cid, {})["anchor_terms"] = matched
    return scores


def retrieve(query: str, k: int = 40, network: str = None,
             foundation: str = None, priority_number: int = None,
             candidate_k: int = None):
    """Filter-first HYBRID retrieval (dense + BM25 + RRF + re-anchor rerank).

    Metadata filters are HARD constraints. Dense and sparse rankers each pull a
    wider candidate pool, RRF fuses them, the re-anchor step rewards literal term
    matches, and the top-k are returned. Returns dicts: {document, metadata,
    distance, score}.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured (Streamlit secrets or .env).")

    candidate_k = candidate_k or max(k * 4, 40)
    where = _where(network, foundation, priority_number)
    where_args = (network, foundation, priority_number)
    client = OpenAI(api_key=key)

    dense = _dense_ranking(query, client, where, candidate_k)
    sparse = _sparse_ranking(query, where_args, candidate_k)

    scores, payloads, prov = _rrf_fuse(dense, sparse)
    scores = _reanchor(query, scores, payloads, prov)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for cid, sc in ranked:
        p = payloads[cid]
        pr = prov.get(cid, {})
        # Human-readable "how was this found" signal for the UI.
        found_by = []
        if pr.get("dense_rank"):
            found_by.append("semantic")
        if pr.get("sparse_rank"):
            found_by.append("keyword")
        out.append({
            **p,
            "score": sc,
            "found_by": found_by or ["semantic"],
            "dense_rank": pr.get("dense_rank"),
            "sparse_rank": pr.get("sparse_rank"),
            "anchor_terms": pr.get("anchor_terms", []),
        })
    return out


def context_from_hits(hits) -> str:
    """Join retrieved chunks into the LLM context block."""
    return "\n\n---\n\n".join(h["document"] for h in hits)
