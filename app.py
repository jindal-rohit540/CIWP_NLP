import os
import re
import html
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CIWP Intelligence", page_icon="🏫", layout="wide")

EMBEDDED_DATA_FILES = {
    "Network 6":  "Data/N6 CIWP_Priority_Extract_20260608_1625.xlsx",
    "Network 10": "Data/N10 CIWP_Priority_Extract_20260608_1629.xlsx",
}

PRIORITY_ORDER = [
    "Effective Instruction",
    "Systems for Student Experience",
    "Connectedness & Wellbeing",
    "Partnerships & Engagement",
]

COL_MAP = {
    "school":       "Plan Name",
    "priority":     "Priority Name",
    "problem":      "Student-Centered Problem",
    "root_cause":   "Root Cause (adult-facing)",
    "toa_if":       "ToA: If We",
    "toa_then":     "ToA: Then We See",
    "toa_leads":    "ToA: Which Leads To",
    "goal_y1":      "Year 1 Practice Goal",
    "goal_y1_tgt":  "Year 1 Practice Goal Target",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_html(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)
    return df


@st.cache_data
def load_embedded_data() -> pd.DataFrame:
    frames = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for net, path in EMBEDDED_DATA_FILES.items():
        full = os.path.join(script_dir, path)
        if os.path.exists(full):
            df = pd.read_excel(full)
            df["Network"] = net
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _decode_html(pd.concat(frames, ignore_index=True))


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
    return _decode_html(pd.concat(frames, ignore_index=True))


def clean(text) -> str:
    return str(text).strip() if pd.notna(text) else ""


def school_name(plan_name: str) -> str:
    """Strip the ' CIWP Cycle YYYY-YYYY' suffix."""
    return re.sub(r"\s+CIWP Cycle.*$", "", plan_name, flags=re.IGNORECASE).strip()


def build_context(rows: pd.DataFrame, max_rows: int = 60) -> str:
    """Build a compact text context from the filtered rows."""
    if len(rows) > max_rows:
        rows = rows.iloc[::2]  # take every other row to stay within token budget
    parts = []
    for _, r in rows.iterrows():
        parts.append(
            f"SCHOOL: {school_name(clean(r.get(COL_MAP['school'], '')))}\n"
            f"NETWORK: {clean(r.get('Network', ''))}\n"
            f"PRIORITY: {clean(r.get(COL_MAP['priority'], ''))}\n"
            f"SCP: {clean(r.get(COL_MAP['problem'], ''))}\n"
            f"ROOT CAUSE: {clean(r.get(COL_MAP['root_cause'], ''))}\n"
            f"TOA-IF: {clean(r.get(COL_MAP['toa_if'], ''))}\n"
            f"TOA-THEN: {clean(r.get(COL_MAP['toa_then'], ''))}\n"
            f"TOA-LEADS: {clean(r.get(COL_MAP['toa_leads'], ''))}\n"
            f"GOAL Y1: {clean(r.get(COL_MAP['goal_y1'], ''))}\n"
            "---"
        )
    return "\n".join(parts)


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


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🏫 CIWP Intelligence — CPS Network Analyzer")
    st.caption("Analyze Continuous Improvement Work Plans across Chicago Public Schools networks.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data")
        uploaded_files = st.file_uploader(
            "Upload additional network extract(s)",
            type=["xlsx"],
            accept_multiple_files=True,
            help="Upload CIWP Priority Extract .xlsx files for any network. "
                 "Networks 6 and 10 are already loaded.",
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
        networks = ["All Networks"] + sorted(df["Network"].dropna().unique().tolist())
        sel_net = st.selectbox("Network", networks)

        priorities = ["All Priorities"] + PRIORITY_ORDER
        sel_pri = st.selectbox("Priority Area", priorities)

        keyword = st.text_input("Keyword search", placeholder="e.g. MTSS, coaching, math")

        st.divider()
        st.header("Model")
        model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])

    # Apply filters
    filt = df.copy()
    if sel_net != "All Networks":
        filt = filt[filt["Network"] == sel_net]
    if sel_pri != "All Priorities":
        filt = filt[filt[COL_MAP["priority"]] == sel_pri]
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
            grouped = filt.groupby(COL_MAP["priority"], sort=False)
            for pri in PRIORITY_ORDER:
                if pri not in grouped.groups:
                    continue
                grp = grouped.get_group(pri)
                n_sch = grp[COL_MAP["school"]].nunique()
                with st.expander(f"**{pri}** — {n_sch} schools, {len(grp)} plans", expanded=False):
                    for _, row in grp.iterrows():
                        sname = school_name(clean(row.get(COL_MAP["school"], "")))
                        net   = clean(row.get("Network", ""))
                        st.markdown(f"#### {sname} `{net}`")
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
                            st.markdown("**Year 1 Goal**")
                            st.write(clean(row.get(COL_MAP["goal_y1"], "")))
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
            if st.button("Ask", type="primary", disabled=not query):
                ctx = build_context(filt)
                n_sch = len(schools)
                system_prompt = (
                    "You are an expert education analyst for Chicago Public Schools. "
                    "You have been given CIWP plan data for multiple schools. "
                    "Answer the user's question using ONLY the data provided below. "
                    "Be specific — name schools and quote exact language when relevant. "
                    f"There are {n_sch} schools in this dataset.\n\n"
                    f"DATA:\n{ctx}"
                )
                with st.spinner("Analyzing plans…"):
                    answer = call_openai(system_prompt, query, model=model)
                st.markdown("### Answer")
                st.markdown(answer)

    # ── Tab 4: Trend Report ───────────────────────────────────────────────────
    with tab_trend:
        st.markdown(
            "One-click structured trend report for CO directors. "
            "Identifies network-wide themes, PD gaps, and alignment strengths."
        )
        n_sch = len(schools)
        net_label = sel_net if sel_net != "All Networks" else "All Networks"
        pri_label = sel_pri if sel_pri != "All Priorities" else "All Foundations"

        if st.button("Generate Trend Report", type="primary"):
            ctx = build_context(filt)
            system_prompt = (
                "You are a senior education analyst for Chicago Public Schools writing "
                "a structured network-level trend report for Central Office directors. "
                "Use markdown headers and bullet points. Be specific — cite school names, "
                "quote exact plan language, and provide counts. "
                f"Dataset: {net_label}, {pri_label}, {n_sch} schools, {n_plans} priority plans.\n\n"
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
            st.download_button(
                "Download Report (Markdown)",
                data=report,
                file_name=f"CIWP_Trend_Report_{net_label}_{pri_label}.md".replace(" ", "_"),
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
