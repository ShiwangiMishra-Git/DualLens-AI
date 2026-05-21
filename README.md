# DualLens-AI

> **Dual-lens investment intelligence** — quantitative financial metrics meet qualitative AI initiative analysis, ranked deterministically across three investor strategies.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.3.4-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

[![View Notebook](https://img.shields.io/badge/nbviewer-View%20Executed%20Notebook-orange?logo=jupyter)](https://nbviewer.org/github/ShiwangiMishra-Git/DualLens-AI/blob/master/notebooks/DualLens_Analytics_executed.ipynb)
[![HTML Report](https://img.shields.io/badge/report-HTML%20Export-blueviolet)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ShiwangiMishra-Git/DualLens-AI/master/reports/DualLens_Analytics_executed.html)
[![HLD](https://img.shields.io/badge/docs-HLD-informational)](docs/HLD.md)
[![LLD](https://img.shields.io/badge/docs-LLD-informational)](docs/LLD.md)

An investment ranking system for five major tech companies (MSFT, GOOGL, AMZN, NVDA, IBM) that combines **quantitative financial analysis** with **qualitative AI initiative analysis** using RAG, LLM signal extraction, and deterministic Python scoring.

---

## Overview

DualLens-AI answers: *"Which tech company is the best investment given its AI initiatives and financial performance?"*

It does so across three investor profiles — **Conservative**, **Balanced**, and **Growth** — using two complementary approaches:

| Approach | Description |
|---|---|
| **Section 5A — LLM Scoring** | Single LLM prompt assigns numeric scores and writes a ranked narrative |
| **Section 5B — Deterministic** | LLM extracts 8 AI signals; Python computes all scores deterministically |

**Recommended approach: Section 5B.** Rankings are fully reproducible — identical inputs always produce identical rankings, regardless of how many times the notebook is run.

---

## Problem Statement

Five company PDFs exceeded the model's context window. A RAG pipeline with map-reduce summarisation compresses each PDF into a per-company AI summary used for signal extraction and ranking.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                      DATA INPUTS                        │
  └──────────────┬──────────────────────────┬───────────────┘
                 │                          │
  ┌──────────────▼──────────┐  ┌────────────▼──────────────┐
  │  Financial Data         │  │  AI Initiatives (PDFs)    │
  │  yfinance — 5 metrics   │  │  5 company PDFs           │
  └──────────────┬──────────┘  └────────────┬──────────────┘
                 │                          │
                 │             ┌────────────▼──────────────┐
                 │             │  RAG Pipeline             │
                 │             │  ChromaDB + map-reduce    │
                 │             │  → per-company AI summary │
                 │             └────────────┬──────────────┘
                 │                          │
                 │             ┌────────────▼──────────────┐
                 │             │  LLM Signal Extractor     │
                 │             │  gpt-4o-mini              │
                 │             │  8 signals × 5 companies  │
                 │             │  Output: none/partial/full│
                 │             └────────────┬──────────────┘
                 │                          │
                 └──────────────┬───────────┘
                                ▼
  ┌─────────────────────────────────────────────────────────┐
  │           Python Scoring Engine (Deterministic)         │
  │   AI Score  = Σ (signal_weight × SIGNAL_LEVELS[label]) │
  │   Fin Score = Σ (rank_score × strategy_weight)         │
  │   Total     = AI Score + Financial Score                │
  └─────────────────────────────────────────────────────────┘
```

---

## Pipeline Workflow

1. **Financial Analysis** — 3-year price history and 5 metrics via yfinance; rank-based scoring per strategy
2. **RAG Pipeline** — PDF chunks embedded in ChromaDB; investment questions answered with retrieved evidence
3. **Map-Reduce Summarisation** — per-company AI summary compressed to fit LLM context window; cached to disk
4. **LLM Scoring (5A)** — single LLM call assigns scores and writes a ranked narrative
5. **Deterministic Scoring (5B)** — LLM classifies 8 signals; Python computes all scores; recommended approach
6. **Evaluation** — three `gpt-4o` judge calls: signal groundedness, explanation consistency, relevance

---

## AI Signals Extracted (Section 5B)

| Signal | Investment Meaning |
|---|---|
| `core_business_alignment` | Direct revenue impact — AI drives primary business model |
| `enterprise_ai` | Recurring enterprise revenue — B2B platform and SaaS contracts |
| `public_release` | Commercialisation maturity — product is live, not just announced |
| `hardware_ai` | Silicon ownership — proprietary AI chips in production enable vertical integration and defensible margin |
| `consumer_ai` | Market reach — consumer data advantage and engagement at scale |
| `novel_research` | Future growth — research pipeline signals next-generation products |
| `risk_mitigation_present` | Governance quality — proactive risk management reduces regulatory risk |
| `infrastructure_moat` | Structural moat — foundational AI compute dependency (cloud platforms e.g. Azure/AWS score partial; de facto compute standard e.g. CUDA scores full) creates industry-wide switching costs |

---

## Key Results (Section 5B — Deterministic, Latest Run)

| Strategy | Rank 1 | Score | Rank 2 | Score | Rank 3 | Score |
|---|---|---|---|---|---|---|
| Conservative | MSFT | 3.47/4.5 Buy | GOOGL | 3.19/4.5 Hold | AMZN | 2.88/4.5 Hold |
| Balanced | MSFT | 3.49/4.5 Buy | GOOGL | 3.26/4.5 Buy | AMZN | 3.01/4.5 Hold |
| Growth | MSFT | 3.41/4.5 Buy | GOOGL | 3.28/4.5 Buy | AMZN | 2.96/4.5 Hold |

**NVIDIA (NVDA):** *Hold* under Growth (`hardware_ai=full`, `infrastructure_moat=full` — CUDA dominance contributes 1.10 pts at growth weights); *Sell* under Conservative and Balanced where zero enterprise/consumer signals dominate.

**Robustness:** Weight Sensitivity Analysis confirms rankings are stable across all ±20% weight shifts.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ShiwangiMishra-Git/DualLens-AI.git
cd DualLens-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

Create a `config.json` file in the project root:

```json
{
    "API_KEY": "your_openai_api_key_here",
    "OPENAI_API_BASE": "https://your_openai_api_base/v1"
}
```

> Tested against the Great Learning OpenAI-compatible proxy using `gpt-4o-mini` for the pipeline and `gpt-4o` for evaluation.

### 4. Run the notebook

Open `notebooks/DualLens_Analytics.ipynb` and run all cells top-to-bottom.

- The PDF zip is automatically extracted to `data/Companies-AI-Initiatives/` on first run
- Generated summaries, signal extractions, and explanations are cached in `cache/` — subsequent runs skip expensive LLM calls
- The stock price chart is saved to `outputs/`

---

## Evaluation Framework

Three LLM-as-Judge (`gpt-4o`) evaluations are run at the end of the notebook:

| Evaluation | What It Checks | Score |
|---|---|---|
| Signal Groundedness | Every signal classification is traceable to text in the AI summary | 5/5 |
| Explanation Consistency | The LLM narrative references only pre-computed scores and rankings | 5/5 |
| Explanation Relevance | The explanation directly answers the investment question | 5/5 |

The RAG pipeline is separately evaluated for Groundedness (5/5) and Relevance (5/5) across six test questions, including a correct scope-refusal for an out-of-scope company.

---

## Project Structure

```
DualLens-AI/
├── README.md                              ← this file
├── requirements.txt                       ← pinned dependencies
├── .gitignore
├── config.json                            ← API credentials (create locally; gitignored)
├── LICENSE                                ← MIT
├── notebooks/
│   ├── DualLens_Analytics.ipynb          ← clean notebook (run top-to-bottom)
│   └── DualLens_Analytics_executed.ipynb ← pre-executed with all outputs (view on nbviewer)
├── reports/
│   └── DualLens_Analytics_executed.html  ← pre-executed HTML export
├── data/
│   ├── Companies-AI-Initiatives.zip      ← source PDFs (auto-extracted at runtime)
│   └── Companies-AI-Initiatives/         ← extracted PDFs (GOOGL MSFT IBM NVDA AMZN)
├── docs/
│   ├── HLD.md                             ← High-Level Design
│   └── LLD.md                             ← Low-Level Design
├── cache/                                 ← generated at runtime; gitignored
└── outputs/                               ← generated at runtime; gitignored
```

---

## Limitations

- **Static PDFs** — AI initiative summaries are derived from fixed documents; rankings go stale as companies announce new products
- **Five companies only** — ChromaDB is in-memory; scaling to more companies requires persisting the vector store
- **Proxy API** — tested against Great Learning's OpenAI-compatible proxy; behaviour may differ on the official OpenAI API
- **Financial metrics** — five metrics (Market Cap, P/E, Dividend Yield, Beta, Total Revenue) do not capture forward-looking indicators like revenue growth or free cash flow

---

## Future Scope

| Enhancement | What It Enables |
|---|---|
| Persist ChromaDB to disk | Scale beyond 5 companies without re-embedding on every run |
| Add `revenueGrowth`, `forwardPE`, `freeCashflow` | Distinguish profitable growers from large-but-stagnant companies |
| Streamlit / Gradio UI | Make rankings accessible to non-technical investors |
| Real-time news + earnings transcripts | Keep rankings current beyond static PDFs |
| Agentic architecture | Orchestrator dynamically selects tools; reflects and retries on low-confidence results |

---

## Key Takeaways

- **RAG enables qualitative retrieval** — company AI initiative PDFs are chunked, embedded, and retrieved with company-aware metadata filtering
- **Deterministic scoring improves reproducibility** — confining the LLM to 3-label signal classification and computing all arithmetic in Python eliminates run-to-run variance
- **Strategy-specific weighting changes outcomes** — NVDA flips from Sell to Hold under Growth weights, where `hardware_ai` and `infrastructure_moat` dominate
- **Evaluation confirms quality** — LLM-as-Judge scores all three dimensions at 5/5: signal groundedness, explanation consistency, and relevance

---

## Disclaimer

Rankings and recommendations are generated by an AI system and **do not constitute financial advice**. Do not make investment decisions based on these outputs.

---

*Built as part of the Johns Hopkins University Agentic AI curriculum.*
