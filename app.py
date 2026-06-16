import os
import re
import html
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

import rag

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CIWP Intelligence", page_icon="🏫", layout="wide")

# Authoritative full-district extract (5,509 rows · 580 schools · all networks).
EMBEDDED_DATA_FILE = "Data/CIWP_Priority_Extract_20260612_1529.xlsx"

# Base foundations (Priority Name carries a "Priority N: " prefix we strip for filtering).
PRIORITY_ORDER = [
    "Effective Instruction",
    "Systems for Student Experience",
    "Connectedness & Wellbeing",
    "Partnerships & Engagement",
    "Postsecondary Success",
]

# Legacy cap, retained only for any non-RAG fallback path.
MAX_CONTEXT_ROWS = 150

# RAG retrieval: how many (School x Priority) chains to pull into the LLM context.
# Filters act as hard metadata constraints; these are the semantic top-k within scope.
RETRIEVE_K_QUERY = 30   # focused Q&A
RETRIEVE_K_TREND = 60   # broader trend-report coverage

COL_MAP = {
    "school":         "Plan Name",
    "priority":       "Priority Name",
    "foundation":     "Foundation",          # derived: Priority Name with the "Priority N:" prefix stripped
    "problem":        "Student-Centered Problem",
    "root_cause":     "Root Cause (adult-facing)",
    "toa_if":         "ToA: If We",
    "toa_then":       "ToA: Then We See",
    "toa_leads":      "ToA: Which Leads To",
    "practice_desc":  "Practice Description",
    "goal_y1":        "Year 1 Practice Goal",
    "goal_y1_tgt":    "Year 1 Practice Goal Target",
    "goal_y2":        "Year 2 Practice Goal",
    "goal_y2_tgt":    "Year 2 Practice Goal Target",
    "goal_y3":        "Year 3 Practice Goal",
    "goal_y3_tgt":    "Year 3 Practice Goal Target",
    "perf_goal":      "Performance Goal",
    "metric":         "Metric",
    "student_group":  "Student Group",
}


def derive_foundation(priority_name) -> str:
    """Strip the 'Priority N: ' disambiguation prefix to get the base foundation."""
    return re.sub(r"^\s*Priority\s*\d+\s*:\s*", "", str(priority_name)).strip()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_html(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Common post-load steps: decode HTML entities and derive the Foundation column."""
    df = _decode_html(df)
    if COL_MAP["priority"] in df.columns:
        df[COL_MAP["foundation"]] = df[COL_MAP["priority"]].apply(derive_foundation)
    return df


@st.cache_data
def load_embedded_data() -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(script_dir, EMBEDDED_DATA_FILE)
    if not os.path.exists(full):
        return pd.DataFrame()
    df = pd.read_excel(full)  # the full-district extract carries its own Network column
    return _finalize(df)


def load_uploaded_data(uploaded_files) -> pd.DataFrame:
    frames = []
    for uf in uploaded_files:
        df = pd.read_excel(uf)
        # Use the Network column from the file if present, else derive from filename
        if "Network" not in df.columns:
            df["Network"] = uf.name.replace(".xlsx", "")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _finalize(pd.concat(frames, ignore_index=True))


def clean(text) -> str:
    return str(text).strip() if pd.notna(text) else ""


def school_name(plan_name: str) -> str:
    """Strip the ' CIWP Cycle YYYY-YYYY' suffix."""
    return re.sub(r"\s+CIWP Cycle.*$", "", plan_name, flags=re.IGNORECASE).strip()


def sort_networks(values) -> list:
    """Sort 'Network N' numerically, then non-numeric labels (Charter, ISP, Options) alphabetically."""
    def key(v):
        m = re.match(r"Network\s+(\d+)", str(v))
        return (0, int(m.group(1)), "") if m else (1, 0, str(v))
    return sorted(values, key=key)


def build_context(rows: pd.DataFrame, max_rows: int = MAX_CONTEXT_ROWS):
    """Build a compact text context from the *already-filtered* rows.

    The full dataset (5,509 rows) never fits in a prompt, so callers filter
    first and we hard-cap here. Returns (context_text, n_rows_used, total_rows)
    so the UI can warn when only a sample was sent.
    """
    total = len(rows)
    if total > max_rows:
        # Even spacing across the filtered set so the sample isn't front-loaded.
        step = max(1, total // max_rows)
        rows = rows.iloc[::step].head(max_rows)
    used = len(rows)
    parts = []
    for _, r in rows.iterrows():
        parts.append(
            f"SCHOOL: {school_name(clean(r.get(COL_MAP['school'], '')))}\n"
            f"NETWORK: {clean(r.get('Network', ''))}\n"
            f"PRIORITY: {clean(r.get(COL_MAP['priority'], ''))}\n"
            f"STUDENT GROUP: {clean(r.get(COL_MAP['student_group'], ''))}\n"
            f"SCP: {clean(r.get(COL_MAP['problem'], ''))}\n"
            f"ROOT CAUSE: {clean(r.get(COL_MAP['root_cause'], ''))}\n"
            f"TOA-IF: {clean(r.get(COL_MAP['toa_if'], ''))}\n"
            f"TOA-THEN: {clean(r.get(COL_MAP['toa_then'], ''))}\n"
            f"TOA-LEADS: {clean(r.get(COL_MAP['toa_leads'], ''))}\n"
            f"PRACTICE GOAL Y1: {clean(r.get(COL_MAP['goal_y1'], ''))} (target: {clean(r.get(COL_MAP['goal_y1_tgt'], ''))})\n"
            f"PRACTICE GOAL Y2: {clean(r.get(COL_MAP['goal_y2'], ''))}\n"
            f"PRACTICE GOAL Y3: {clean(r.get(COL_MAP['goal_y3'], ''))}\n"
            f"PERFORMANCE GOAL: {clean(r.get(COL_MAP['perf_goal'], ''))}\n"
            f"METRIC: {clean(r.get(COL_MAP['metric'], ''))}\n"
            "---"
        )
    return "\n".join(parts), used, total


def call_openai(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    # On Streamlit Cloud the key comes from st.secrets; locally it comes from .env
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "Error: OPENAI_API_KEY not configured. Add it to Streamlit secrets or your .env file."
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            return (
                "**Rate limit reached.** Your OpenAI account has hit its usage quota. "
                "Please add billing credits at platform.openai.com/settings/billing, "
                "or try switching to **gpt-4o-mini** (cheaper) in the sidebar."
            )
        if "auth" in err.lower() or "401" in err:
            return "**Invalid API key.** Check your key in Streamlit secrets or .env."
        return f"**API error:** {err}"


# ── Evaluation logic (deterministic + LLM) ───────────────────────────────────
def evaluate_row(row: pd.Series) -> dict:
    """Run rule-based checks on a single school row."""
    scp = clean(row.get(COL_MAP["problem"], ""))
    rc  = clean(row.get(COL_MAP["root_cause"], ""))
    toa_if    = clean(row.get(COL_MAP["toa_if"], ""))
    toa_then  = clean(row.get(COL_MAP["toa_then"], ""))
    toa_leads = clean(row.get(COL_MAP["toa_leads"], ""))
    goal      = clean(row.get(COL_MAP["goal_y1"], ""))

    # SCP: cause-free check
    causal_words = ["because", "due to", "as a result", "since teachers", "since staff"]
    scp_cause_free = not any(w in scp.lower() for w in causal_words)

    # SCP: receipt check (contains a percentage or number)
    scp_has_receipt = bool(re.search(r"\d+%|\d+\s*(students|schools|teachers)", scp, re.I))

    # RC: locus of control — flag student/external blame
    external_signals = ["poverty", "home life", "parental", "parent involvement",
                        "district budget", "student motivation", "attendance"]
    adult_signals    = ["as adults", "we utilize", "we have not", "leadership",
                        "principal", "coach", "professional learning", "adult"]
    rc_external = any(w in rc.lower() for w in external_signals)
    rc_adult    = any(w in rc.lower() for w in adult_signals)
    if rc_external:
        rc_locus = "Deficit-Thinking ⚠️"
    elif rc_adult:
        rc_locus = "Leadership-Facing ✅"
    else:
        rc_locus = "Unclear 🔶"

    # ToA: structure check
    toa_complete = bool(toa_if and toa_then and toa_leads)

    # Goal: exists
    goal_present = bool(goal)

    return {
        "school":          school_name(clean(row.get(COL_MAP["school"], ""))),
        "network":         clean(row.get("Network", "")),
        "priority":        clean(row.get(COL_MAP["priority"], "")),
        "scp_cause_free":  "✅" if scp_cause_free else "⚠️ Has causal language",
        "scp_receipt":     "✅" if scp_has_receipt else "⚠️ Missing data metrics",
        "rc_locus":        rc_locus,
        "toa_complete":    "✅" if toa_complete else "⚠️ Incomplete structure",
        "goal_present":    "✅" if goal_present else "⚠️ Missing",
    }


def render_sources(hits, label="Retrieved sources"):
    """Show exactly which CIWP logic chains were retrieved and HOW (provenance).

    Surfaces, per chunk: the school/network/priority, the fusion score, which
    ranker(s) found it (semantic vs. keyword) with their ranks, and any literal
    query terms that triggered the re-anchor boost — plus the raw chunk text.
    """
    with st.expander(f"📑 {label} — {len(hits)} CIWP logic chains used", expanded=False):
        st.caption(
            "These are the exact plan excerpts the answer is grounded in. "
            "`semantic` = matched by meaning (embeddings); `keyword` = matched by "
            "literal term (BM25); **bold terms** triggered the re-anchor boost."
        )
        # Compact ranked table first.
        table = []
        for i, h in enumerate(hits, 1):
            m = h["metadata"]
            table.append({
                "#": i,
                "School": m.get("school", ""),
                "Network": m.get("network", ""),
                "Priority": m.get("priority_name", ""),
                "Found by": " + ".join(h.get("found_by", [])),
                "Score": round(h.get("score", 0.0), 4),
                "Sem. rank": h.get("dense_rank") or "—",
                "Kw. rank": h.get("sparse_rank") or "—",
                "Anchor terms": ", ".join(h.get("anchor_terms", [])) or "—",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        # Full text per source.
        for i, h in enumerate(hits, 1):
            m = h["metadata"]
            with st.expander(
                f"{i}. {m.get('school','')} · {m.get('network','')} · {m.get('priority_name','')}",
                expanded=False,
            ):
                signals = " + ".join(h.get("found_by", []))
                bits = [f"**Found by:** {signals}", f"**Fusion score:** {h.get('score',0):.4f}"]
                if h.get("dense_rank"):
                    bits.append(f"semantic rank #{h['dense_rank']}")
                if h.get("sparse_rank"):
                    bits.append(f"keyword rank #{h['sparse_rank']}")
                if h.get("anchor_terms"):
                    bits.append("anchored on: " + ", ".join(f"**{t}**" for t in h["anchor_terms"]))
                st.markdown(" · ".join(bits))
                st.code(h["document"], language=None)


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🏫 CIWP Intelligence — CPS District Analyzer")
    st.caption("Analyze Continuous Improvement Work Plans across all Chicago Public Schools networks.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data")
        uploaded_files = st.file_uploader(
            "Upload additional network extract(s)",
            type=["xlsx"],
            accept_multiple_files=True,
            help="Upload CIWP Priority Extract .xlsx files for any network. "
                 "The full district extract (all networks) is already loaded.",
        )

        st.divider()
        st.header("Filters")

    # Load and merge embedded + uploaded data
    embedded = load_embedded_data()
    if uploaded_files:
        uploaded = load_uploaded_data(uploaded_files)
        df = pd.concat([embedded, uploaded], ignore_index=True) if not embedded.empty else uploaded
    else:
        df = embedded

    if df.empty:
        st.warning("No data loaded. Upload a CIWP Priority Extract .xlsx to get started.")
        return

    with st.sidebar:
        networks = ["All Networks"] + sort_networks(df["Network"].dropna().unique().tolist())
        sel_net = st.selectbox("Network", networks)

        # Show only foundations present in the data, ordered by PRIORITY_ORDER then any extras.
        present = set(df[COL_MAP["foundation"]].dropna().unique())
        ordered = [p for p in PRIORITY_ORDER if p in present]
        extras = sorted(present - set(PRIORITY_ORDER))
        priorities = ["All Priorities"] + ordered + extras
        sel_pri = st.selectbox("Priority Area", priorities)

        keyword = st.text_input("Keyword search", placeholder="e.g. MTSS, coaching, math")

        st.divider()
        st.header("Model")
        model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])

        st.divider()
        st.header("Search Index")

        # Auto-build once per container: Streamlit Cloud's disk is ephemeral, so on
        # a cold start the index is empty. Build it automatically (guarded so it
        # runs once) instead of forcing a stakeholder to click a button.
        status = rag.index_status()
        if not status.get("count") and not st.session_state.get("_index_attempted"):
            st.session_state["_index_attempted"] = True
            bar = st.progress(0.0, text="Preparing search index (first load only)…")
            try:
                n = rag.build_index(
                    df, upsert=False,
                    progress=lambda d, t: bar.progress(d / t, text=f"Embedding {d}/{t} chains…"),
                )
                bar.progress(1.0, text=f"Indexed {n} chains.")
                status = rag.index_status()
            except Exception as e:
                bar.empty()
                st.warning(f"Auto-build skipped: {e}")

        if status.get("count"):
            st.caption(f"✅ {status['count']} CIWP logic chains indexed for semantic search.")
        else:
            st.caption("⚠️ No search index yet. Build it to enable semantic AI Query & Trend Report.")

        if st.button("Build / refresh index"):
            bar = st.progress(0.0, text="Embedding CIWP logic chains…")
            try:
                n = rag.build_index(
                    df, upsert=False,
                    progress=lambda d, t: bar.progress(d / t, text=f"Embedding {d}/{t} chains…"),
                )
                bar.progress(1.0, text=f"Indexed {n} chains.")
                st.success(f"Indexed {n} CIWP logic chains.")
            except Exception as e:
                bar.empty()
                st.error(f"Index build failed: {e}")

    index_ready = bool(rag.index_status().get("count"))

    # Apply filters
    filt = df.copy()
    if sel_net != "All Networks":
        filt = filt[filt["Network"] == sel_net]
    if sel_pri != "All Priorities":
        filt = filt[filt[COL_MAP["foundation"]] == sel_pri]
    if keyword:
        kw = keyword.lower()
        mask = filt.apply(
            lambda r: any(kw in str(v).lower() for v in r.values), axis=1
        )
        filt = filt[mask]

    schools = filt[COL_MAP["school"]].dropna().apply(school_name).unique()
    n_plans = len(filt)
    st.markdown(
        f"**{len(schools)} schools · {n_plans} priority plans** — "
        f"{sel_net} · {sel_pri}"
        + (f' · keyword: "{keyword}"' if keyword else "")
    )

    tab_browse, tab_audit, tab_query, tab_trend = st.tabs(
        ["📋 Browse", "🔍 Audit", "💬 AI Query", "📊 Trend Report"]
    )

    # ── Tab 1: Browse ─────────────────────────────────────────────────────────
    with tab_browse:
        if filt.empty:
            st.info("No records match the current filters.")
        else:
            # Group by base Foundation (Priority Name carries a "Priority N:" prefix).
            grouped = filt.groupby(COL_MAP["foundation"], sort=False)
            present = [p for p in PRIORITY_ORDER if p in grouped.groups]
            present += [p for p in grouped.groups if p not in PRIORITY_ORDER]
            for pri in present:
                grp = grouped.get_group(pri)
                n_sch = grp[COL_MAP["school"]].nunique()
                with st.expander(f"**{pri}** — {n_sch} schools, {len(grp)} plans", expanded=False):
                    for _, row in grp.iterrows():
                        sname = school_name(clean(row.get(COL_MAP["school"], "")))
                        net   = clean(row.get("Network", ""))
                        full_pri = clean(row.get(COL_MAP["priority"], ""))
                        sg = clean(row.get(COL_MAP["student_group"], ""))
                        st.markdown(f"#### {sname} `{net}`")
                        meta = f"*{full_pri}*" + (f" · **Student Group:** {sg}" if sg else "")
                        st.markdown(meta)
                        cols = st.columns([1, 1])
                        with cols[0]:
                            st.markdown("**Student-Centered Problem**")
                            st.write(clean(row.get(COL_MAP["problem"], "")))
                            st.markdown("**Root Cause**")
                            st.write(clean(row.get(COL_MAP["root_cause"], "")))
                        with cols[1]:
                            st.markdown("**Theory of Action**")
                            st.write(f"*If we...* {clean(row.get(COL_MAP['toa_if'], ''))}")
                            st.write(f"*Then we see...* {clean(row.get(COL_MAP['toa_then'], ''))}")
                            st.write(f"*Which leads to...* {clean(row.get(COL_MAP['toa_leads'], ''))}")

                        # Practice goals (Y1–Y3) and Performance goal — many rows have only some of these.
                        g1, g1t = clean(row.get(COL_MAP["goal_y1"], "")), clean(row.get(COL_MAP["goal_y1_tgt"], ""))
                        g2 = clean(row.get(COL_MAP["goal_y2"], ""))
                        g3 = clean(row.get(COL_MAP["goal_y3"], ""))
                        pg = clean(row.get(COL_MAP["perf_goal"], ""))
                        metric = clean(row.get(COL_MAP["metric"], ""))
                        if any([g1, g2, g3]):
                            st.markdown("**Practice Goals**")
                            if g1:
                                st.write(f"*Year 1:* {g1}" + (f"  ·  **Target:** {g1t}" if g1t else ""))
                            if g2:
                                st.write(f"*Year 2:* {g2}")
                            if g3:
                                st.write(f"*Year 3:* {g3}")
                        if pg or metric:
                            st.markdown("**Performance Goal**")
                            if pg:
                                st.write(pg)
                            if metric:
                                st.caption(f"Metric: {metric}")
                        st.divider()

    # ── Tab 2: Audit ──────────────────────────────────────────────────────────
    with tab_audit:
        st.markdown(
            "Rule-based alignment check for every school against CPS CIWP quality indicators. "
            "No AI call needed — instant results."
        )
        if filt.empty:
            st.info("No records match the current filters.")
        else:
            results = [evaluate_row(row) for _, row in filt.iterrows()]
            audit_df = pd.DataFrame(results).rename(columns={
                "school":         "School",
                "network":        "Network",
                "priority":       "Priority",
                "scp_cause_free": "SCP: Cause-Free?",
                "scp_receipt":    "SCP: Has Data?",
                "rc_locus":       "RC: Locus of Control",
                "toa_complete":   "ToA: Complete?",
                "goal_present":   "Y1 Goal: Present?",
            })
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

            # Summary metrics
            total = len(results)
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SCP Cause-Free", f"{sum('✅' in r['scp_cause_free'] for r in results)}/{total}")
            c2.metric("SCP Has Data",   f"{sum('✅' in r['scp_receipt'] for r in results)}/{total}")
            c3.metric("RC Leadership",  f"{sum('✅' in r['rc_locus'] for r in results)}/{total}")
            c4.metric("ToA Complete",   f"{sum('✅' in r['toa_complete'] for r in results)}/{total}")

    # ── Tab 3: AI Query ───────────────────────────────────────────────────────
    with tab_query:
        st.markdown(
            "Ask any plain-English question about the filtered plans. "
            "The answer is grounded in the actual plan text — not the model's training data."
        )
        col_ex, col_inp = st.columns([1, 2])
        with col_ex:
            st.markdown("**Example questions**")
            examples = [
                "What are the most common root causes?",
                "Which schools focus on math outcomes?",
                "Summarize the MTSS strategies used",
                "Which schools mention English Learners in their SCP?",
                "Are any root causes blaming students or external factors?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}"):
                    st.session_state["query_input"] = ex

        with col_inp:
            query = st.text_area(
                "Your question",
                value=st.session_state.get("query_input", ""),
                height=100,
                key="query_text",
            )
            if not index_ready:
                st.info("Build the search index (sidebar) to enable semantic AI Query.")
            if st.button("Ask", type="primary", disabled=not query or not index_ready):
                # Semantic retrieval — sidebar filters become HARD metadata constraints,
                # similarity ranks the most relevant logic chains within that scope.
                with st.spinner("Retrieving the most relevant CIWP logic chains…"):
                    hits = rag.retrieve(
                        query, k=RETRIEVE_K_QUERY,
                        network=sel_net, foundation=sel_pri,
                    )
                if not hits:
                    st.warning("No indexed plans match the active filters. Widen the filters or rebuild the index.")
                else:
                    ctx = rag.context_from_hits(hits)
                    retrieved_schools = sorted({h["metadata"]["school"] for h in hits})
                    st.caption(
                        f"Grounded in {len(hits)} most-relevant logic chains "
                        f"from {len(retrieved_schools)} schools (filtered: {sel_net} · {sel_pri})."
                    )
                    system_prompt = (
                        "You are an expert education analyst for Chicago Public Schools. "
                        "You have been given the CIWP logic chains most relevant to the user's question. "
                        "Answer using ONLY the data provided below. "
                        "Be specific — name schools and quote exact language when relevant. "
                        "If the retrieved chains don't cover the question, say so plainly.\n\n"
                        f"DATA:\n{ctx}"
                    )
                    with st.spinner("Analyzing plans…"):
                        answer = call_openai(system_prompt, query, model=model)
                    st.markdown("### Answer")
                    st.markdown(answer)
                    render_sources(hits, label="Sources for this answer")

    # ── Tab 4: Trend Report ───────────────────────────────────────────────────
    with tab_trend:
        st.markdown(
            "One-click structured trend report for CO directors. "
            "Identifies network-wide themes, PD gaps, and alignment strengths."
        )
        n_sch = len(schools)
        net_label = sel_net if sel_net != "All Networks" else "All Networks"
        pri_label = sel_pri if sel_pri != "All Priorities" else "All Foundations"

        if not index_ready:
            st.info("Build the search index (sidebar) to enable the trend report.")
        elif n_plans > RETRIEVE_K_TREND:
            st.info(
                f"This selection has **{n_plans}** plans. The report draws on the "
                f"**{RETRIEVE_K_TREND}** most representative logic chains in scope. "
                f"Filter to a single network or priority area for tighter analysis."
            )

        if st.button("Generate Trend Report", type="primary", disabled=not index_ready):
            # Broad scoped retrieval: pull representative chains across the filtered set.
            trend_query = (
                f"student-centered problems, root causes, theories of action, and practice goals "
                f"for {pri_label} across {net_label}"
            )
            with st.spinner("Retrieving representative CIWP logic chains…"):
                hits = rag.retrieve(
                    trend_query, k=RETRIEVE_K_TREND,
                    network=sel_net, foundation=sel_pri,
                )
            if not hits:
                st.warning("No indexed plans match the active filters.")
                st.stop()
            ctx = rag.context_from_hits(hits)
            covered = sorted({h["metadata"]["school"] for h in hits})
            coverage = (
                f"This report draws on {len(hits)} CIWP logic chains "
                f"from {len(covered)} schools matching the active filters."
            )
            system_prompt = (
                "You are a senior education analyst for Chicago Public Schools writing "
                "a structured network-level trend report for Central Office directors. "
                "Use markdown headers and bullet points. Be specific — cite school names, "
                "quote exact plan language, and provide counts. "
                f"Dataset: {net_label}, {pri_label}, {n_sch} schools, {n_plans} priority plans. "
                f"{coverage}\n\n"
                f"DATA:\n{ctx}"
            )
            user_prompt = f"""Write a comprehensive trend report with these 8 sections:

1. **Executive Summary** — 3-4 bullet overview of key findings
2. **Common Student-Centered Problems** — recurring themes with school counts
3. **Root Cause Patterns** — categorize by Leadership/Teacher/Deficit-Thinking; flag any concerns
4. **Theory of Action Alignment** — are If/Then/Which leads to chains coherent? Examples of strong vs. weak
5. **Professional Development Needs** — what PD gaps are visible across schools?
6. **Bright Spots** — 2-3 schools with exemplary CIWP logic worth sharing as models
7. **Schools Needing Support** — schools whose plans need the most revision (with specific reasons)
8. **Recommended CO Actions** — 3-5 concrete next steps for network leaders

Be direct and specific. Avoid vague summaries."""

            with st.spinner("Generating trend report (this may take 20–30 seconds)…"):
                report = call_openai(system_prompt, user_prompt, model=model)
            st.markdown(report)
            render_sources(hits, label="Sources for this report")
            st.download_button(
                "Download Report (Markdown)",
                data=report,
                file_name=f"CIWP_Trend_Report_{net_label}_{pri_label}.md".replace(" ", "_"),
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
