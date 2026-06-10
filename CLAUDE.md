# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **CPS CIWP Intelligence tool** built for Chicago Public Schools (CPS) to analyze Continuous Improvement Work Plans (CIWPs) across school networks. The goal is to replace a manually-operated Gemini "Gem" (single-prompt RAG chatbot) with a more robust, LangGraph-based multi-agent system.

**Rohit Jindal** is a data scientist newly joined to CPS working on this project. He built the initial PoC (`ciwp_v7_anthropic.html`) and is iterating toward a production system.

## Domain Vocabulary (Critical to understand before editing any prompt or evaluation logic)

A CIWP follows a strict chain of logic. Every component must be evaluated against this chain:

- **Foundation** — the broad focus area (e.g., Effective Instruction, SSE, C&W, Partnerships)
- **SCP (Student-Centered Problem)** — what student problem exists; must include quantitative data ("the receipt") from ≥2 sources; must be a "Cause-Free Zone" (no "because", "due to", or adult behavior language)
- **RC (Root Cause)** — why the problem exists; must be 100% Leadership/Adult-facing and within the school's locus of control; must never blame students, parents, poverty, or external factors
- **ToA (Theory of Action)** — a three-part conditional: `If we [leadership lever] → Then we see [teacher practice shift] → Which leads to [student outcome]`; the "Which leads to" must mirror and resolve the SCP
- **Year 1 Practice Goal** — a measurable metric for the "Then we see" adult practice shift

## Current System: `ciwp_v7_anthropic.html`

A single self-contained HTML file. No build step, no server, no install — open in browser.

**Architecture:**
- CIWP data from Networks 6 (25 schools) and 10 (34 schools) is embedded as JSON in a `<script type="application/json" id="ciwpdata">` tag (~182 priority plans)
- Filter UI (network, priority area, keyword) drives a Browse tab
- **AI Query tab**: user types plain-English question → calls Anthropic API directly from the browser using `claude-sonnet-4-20250514`
- **Trend Report tab**: one-click → generates an 8-section structured analysis for CO directors
- API key (`KEY` variable, line 124) is hardcoded — **this must be rotated before any sharing or deployment**
- Uses `anthropic-dangerous-direct-browser-access: true` header for direct browser-to-API calls (PoC only, not production-safe)

**Known limitations (from stakeholder feedback):**
- The Gem sometimes omits schools from analysis; once prompted it corrects itself
- Gemini froze when fed district-wide data (all networks)
- NotebookLM gave incorrect school sets
- The Gem accessed outdated data from prior conversations (e.g., surfaced a school from a different network that was geographically proximate historically) — scope leakage is a known problem

## Data

| File | Network | Schools |
|------|---------|---------|
| `Data/N6 CIWP_Priority_Extract_20260608_1625.xlsx` | Network 6 | 25 |
| `Data/N10 CIWP_Priority_Extract_20260608_1629.xlsx` | Network 10 | 34 |

Columns in the extract: School Name, Foundation/Priority, Student-Centered Problem (SCP), Root Cause (RC), Theory of Action (If/Then/Which leads to), Year 1 Practice Goal, Year 1 Target.

Data becomes public by June/July each year — no PII or privacy concerns for Networks 6 and 10 exports.

## Planned Architecture (LangGraph multi-agent)

The Gemini Gem is a single-prompt RAG — it reads all rows in one pass and loses accuracy on rows in the middle (lost-in-the-middle problem). The target system uses a Map-Reduce LangGraph graph:

1. **Parser/Validator Node** — ingest `.xlsx`, standardize columns, flag missing fields
2. **SCP Evaluator Node** — check cause-free language, presence of data metrics, priority student groups
3. **RC Evaluator Node** — classify locus of control: `Leadership-Facing | Teacher-Facing | Deficit-Thinking/External`
4. **ToA Evaluator Node** — verify If/Then/Which structure; check "Which leads to" semantically mirrors the SCP
5. **Goal Connection Node** — check Year 1 goal measures the "Then we see" adult behavior
6. **Synthesis/Reduce Node** — aggregate school-level evaluations into network themes and PD recommendations

Use Pydantic schemas for structured output from each node so the final report is machine-readable (not free-form prose).

## Stakeholder Context

- **Jessica Zapata Gutowski** (Executive Director of Continuous Improvement) — primary business owner; confirmed Options 1 and 2 are high-value for this summer; prefers Option 2 be built first if only one can be started
- **Conor Moloney** — raised data privacy guardrails; PoC should use CPS-approved LLMs for anything beyond the public dataset
- **Craig Gutierrez** (Director of CI) — shared the Gem's instructions; flagged the scope-leakage bug with historical school data
- **Option 3** (referenced in emails) is more sensitive/complex and lower priority

## Model and API

- Current PoC uses `claude-sonnet-4-20250514` via direct browser fetch to `https://api.anthropic.com/v1/messages`
- For production: use CPS-approved LLM endpoints; the hardcoded API key in the HTML must be replaced with a backend proxy
