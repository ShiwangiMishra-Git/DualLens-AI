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
|----------|-------------|
| **Section 5A — LLM Scoring** | Single LLM prompt assigns numeric scores and writes a ranked narrative |
| **Section 5B — Deterministic** | LLM extracts 8 binary signals; Python computes all scores deterministically |

**Recommended approach: Section 5B.** Rankings are fully reproducible — identical inputs always produce identical rankings, regardless of how many times the notebook is run.

---

## Problem Statement

Passing five company PDFs directly to `gpt-4o-mini` for investment ranking exceeded the model's context window (~128k tokens). A RAG pipeline with map-reduce summarization was built to compress PDF content into per-company AI initiative summaries, which are then used for both qualitative analysis and signal extraction.

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

1. **Financial Analysis (Section 3)** — pulls 3-year price history and 5 financial metrics via yfinance; plots closing prices; ranks companies on each metric
2. **RAG Pipeline (Section 4)** — embeds PDF chunks into ChromaDB; answers investment questions with retrieved evidence; evaluated by LLM-as-Judge
3. **Map-Reduce Summarization** — generates a per-company AI initiative summary that fits within the LLM context window
4. **LLM Scoring / Section 5A** — single LLM call assigns scores and writes a ranked narrative per strategy
5. **Deterministic Scoring / Section 5B** — LLM classifies 8 AI signals; Python computes all scores and rankings; LLM writes explanation from pre-computed results
6. **Evaluation** — three gpt-4o judge calls score signal groundedness, explanation consistency, and relevance

---

## AI Signals Extracted (Section 5B)

| Signal | Investment Meaning |
|--------|--------------------|
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
|----------|--------|-------|--------|-------|--------|-------|
| Conservative | MSFT | 3.47/4.5 Buy | GOOGL | 3.19/4.5 Hold | AMZN | 2.88/4.5 Hold |
| Balanced | MSFT | 3.49/4.5 Buy | GOOGL | 3.26/4.5 Buy | AMZN | 3.01/4.5 Hold |
| Growth | MSFT | 3.41/4.5 Buy | GOOGL | 3.28/4.5 Buy | AMZN | 2.96/4.5 Hold |

**NVIDIA (NVDA):** Earns a *Hold* under the Growth strategy (`infrastructure_moat=full`, `hardware_ai=full` — CUDA/GPU dominance contributes 1.10 pts under growth weights) but *Sell* under Conservative and Balanced, where its zero enterprise/consumer AI signals and high Beta/P/E suppress its score.

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

Copy `.env.example` to `config.json` and fill in your API key and base URL:

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
|------------|---------------|-------|
| Signal Groundedness | Every signal classification is traceable to text in the AI summary | 5/5 |
| Explanation Consistency | The LLM narrative references only pre-computed scores and rankings | 3.67/5 |
| Explanation Relevance | The explanation directly answers the investment question | 5/5 |

The RAG pipeline is separately evaluated for Groundedness (5/5) and Relevance (5/5) across six test questions, including a correct scope-refusal for an out-of-scope company.

---

## Project Structure

```
DualLens-AI/
├── README.md                              ← this file
├── requirements.txt                       ← pinned dependencies
├── .gitignore
├── .env.example                           ← credential template
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
|-------------|-----------------|
| Persist ChromaDB to disk | Scale to S&P 500 tech sector without re-embedding on every run |
| Add `revenueGrowth`, `forwardPE`, `freeCashflow` metrics | Distinguish profitable growers from large-but-stagnant companies |
| Streamlit / Gradio UI | Make the tool usable by non-technical investors |
| Real-time news + earnings transcripts | Keep rankings current beyond static PDFs |
| Backtesting against historical data | Validate whether AI-recommended portfolios outperformed the market |
| Agentic architecture | Orchestrator dynamically decides which tools to call; reflects and retries on low-confidence results |

---

## Disclaimer

Rankings and recommendations are generated by an AI system and **do not constitute financial advice**. Do not make investment decisions based on these outputs.

---

*Built as part of the Johns Hopkins University Agentic AI curriculum.*
