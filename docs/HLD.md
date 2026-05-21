# High-Level Design — DualLens-AI

## The Question That Started Everything

It begins with a deceptively simple question:

> *"Among Microsoft, Google, Amazon, NVIDIA, and IBM — which company is the best investment right now, and why?"*

Any investor can Google a stock price. But what makes this question hard is the word *why*. A good answer requires two completely different kinds of evidence working together: cold financial numbers and a qualitative read on where each company is actually headed in AI.

That tension — between quantitative and qualitative — is what DualLens-AI was built to resolve.

---

## The First Attempt — And Why It Failed

The obvious first approach was to gather each company's AI strategy documents, hand them to a language model, and ask for a ranking.

There were two immediate problems.

**The first was size.** Five large PDFs contain far more text than a language model can read at once. There's a hard limit — called a context window — on how much text can fit in a single prompt. Five company documents together blew past that limit.

**The second was consistency.** Even when a condensed version fit, the model would invent precise-sounding scores — "3.74 out of 4.5" — and those numbers would shift slightly every time it ran. Running the same analysis twice gave different rankings. That's not a system you can trust.

Both problems needed solving before the system could be useful.

---

## The Architecture That Solved It

The solution was to break the pipeline into stages — each stage solving one piece of the problem, passing clean output to the next.

```
  Stage 1            Stage 2            Stage 3
  ─────────          ─────────          ─────────
  Read the           Compress           Extract
  PDFs               documents          AI signals
  (5 companies)      into summaries     per company

  Stage 4            Stage 5
  ─────────          ─────────
  Score each         Rank and
  company in         explain the
  Python             results
```

Here is how data moves through the full system:

```
┌───────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                            │
│                                                               │
│  Company AI PDFs ──────────────────────────────────────────┐  │
│  (MSFT, GOOGL, AMZN, NVDA, IBM)                            │  │
│                                                             │  │
│  Yahoo Finance API ─────────────────────────────────────┐  │  │
│  (live market data)                                      │  │  │
└──────────────────────────────────────────────────────────┼──┼──┘
                                                           │  │
                                           ┌───────────────▼──┤
                                           │  FINANCIAL       │
                                           │  ANALYSIS        │
                                           │  5 metrics ranked│
                                           │  per strategy    │
                                           └───────┬──────────┘
                                                   │
                         ┌─────────────────────────▼──────────┐
                         │           RAG PIPELINE             │
                         │                                    │
                         │  Break PDFs into chunks            │
                         │  Store chunks in vector database   │
                         │  Answer questions by retrieving    │
                         │  only relevant chunks              │
                         └─────────────────────────┬──────────┘
                                                   │
                         ┌─────────────────────────▼──────────┐
                         │       MAP-REDUCE SUMMARISER        │
                         │                                    │
                         │  Compresses all chunks into one    │
                         │  paragraph per company             │
                         │  Cached — skips re-processing      │
                         │  on subsequent runs                │
                         └─────────────────────────┬──────────┘
                                                   │
                         ┌─────────────────────────▼──────────┐
                         │       LLM SIGNAL EXTRACTOR         │
                         │                                    │
                         │  Reads the summary per company     │
                         │  Classifies 8 signals as           │
                         │  none / partial / full             │
                         │  Cached — skips on re-run          │
                         └─────────────────────────┬──────────┘
                                                   │
                         ┌─────────────────────────▼──────────┐
                         │    PYTHON SCORING ENGINE           │
                         │                                    │
                         │  AI Score  = signal weights × level│
                         │  Fin Score = rank × metric weight  │
                         │  Total     = AI Score + Fin Score  │
                         │  Ranking   = Python sort (no LLM)  │
                         └─────────────────────────┬──────────┘
                                                   │
                         ┌─────────────────────────▼──────────┐
                         │       EXPLANATION LAYER            │
                         │                                    │
                         │  LLM writes a narrative explaining │
                         │  the pre-computed rankings         │
                         │  Cannot change scores or rankings  │
                         └─────────────────────────┬──────────┘
                                                   │
                         ┌─────────────────────────▼──────────┐
                         │       EVALUATION LAYER             │
                         │                                    │
                         │  A separate "judge" AI (gpt-4o)    │
                         │  scores the system's outputs on    │
                         │  3 dimensions: 1–5 each            │
                         └────────────────────────────────────┘
```

---

## How Each Component Tells Part of the Story

### Component 1 — Financial Analysis

Before touching the AI documents at all, the system anchors itself in hard numbers. It pulls live market data from Yahoo Finance — market cap, revenue, P/E ratio, beta, and dividend yield — for all five companies.

These five metrics are the financial half of the "dual lens." They tell you whether a company is cheap or expensive right now, stable or volatile, returning cash to shareholders or reinvesting aggressively.

The important design choice here is that companies are ranked against each other — not scored in isolation. The top company on each metric earns the highest score; the bottom earns the lowest. This rank-based approach is immune to outliers: one company with an extreme P/E ratio doesn't distort the scores for the others.

The weights applied to each metric shift depending on the investor's profile. A conservative investor cares more about stability (low beta) and income (high dividend yield). A growth investor cares more about scale (high market cap and revenue).

---

### Component 2 — RAG Pipeline

With the financial picture established, the system turns to the qualitative half: what is each company actually building in AI?

The challenge is the document size problem mentioned earlier. Rather than trying to read everything at once, the system uses a technique called **Retrieval-Augmented Generation (RAG)**. The idea is elegant: instead of feeding the model an entire library, build a searchable index of it and only retrieve the pages that matter.

Concretely, this means breaking the five company PDFs into hundreds of small, overlapping text segments — called "chunks" — and storing them in a vector database. Each chunk is also tagged with its company name, so when a question specifically asks about NVIDIA, the retrieval step only searches NVIDIA's chunks. This company-aware filtering prevents Microsoft's content from appearing when you ask about Amazon.

The RAG component also serves a useful secondary role: it can answer direct questions about individual companies with citations. This is where the system can say, "NVIDIA's CUDA ecosystem is described as the foundational compute layer for the AI industry" and point to the exact passage.

---

### Component 3 — Map-Reduce Summariser

RAG is great for answering specific questions, but the scoring engine needs a complete picture of each company — a single, coherent summary that captures everything relevant without exceeding the model's context window.

This is the map-reduce summariser's job. The name comes from a pattern in distributed computing:

- **Map:** Summarise each small chunk independently. These summaries are generated in parallel, which is fast.
- **Reduce:** Combine all the chunk summaries for a company into one final paragraph.

The result is one tight paragraph per company that represents the full content of their AI initiative document without flooding the model's input with thousands of tokens of raw PDF text.

Because this process involves many model calls and takes significant time, the output is saved to disk. Every subsequent run loads the cached summaries in under a second. This makes the system both fast and cost-efficient on repeat runs.

---

### Component 4 — LLM Signal Extractor

With clean, per-company summaries in hand, the system can now extract the qualitative signals that matter for investors.

The extractor reads each company's summary and classifies it on eight investment-relevant questions:

| Signal | What it's asking |
|--------|-----------------|
| Enterprise AI | Does the company sell AI products to businesses? |
| Consumer AI | Does the company have AI products for everyday users? |
| Hardware AI | Does the company make its own AI chips? |
| Public Release | Are the products actually shipped, not just announced? |
| Core Business Alignment | Is AI central to how the company makes money? |
| Novel Research | Is the company doing cutting-edge AI research? |
| Risk Mitigation | Does the company have an AI safety/governance plan? |
| Infrastructure Moat | Is the company's technology what the rest of the AI industry depends on? |

Two signals — **Hardware AI** and **Infrastructure Moat** — were added specifically to avoid underscoring NVIDIA. Standard product signals (`enterprise_ai`, `consumer_ai`, `public_release`) measure how a company deploys AI to customers. NVIDIA has no consumer or enterprise AI products — it builds the infrastructure every other company's AI runs on. Without these two signals, NVIDIA would rank last under every strategy, which misrepresents a company whose GPUs and CUDA ecosystem power the AI initiatives of every other company in the analysis.

Each signal is classified as `none`, `partial`, or `full` — nothing more. This narrow, three-way classification is a deliberate design choice. Asking the model to pick from three options is a much simpler and more reliable task than asking it to invent a number on a continuous scale. The output is short, which means there's less opportunity for the model to drift or hallucinate.

Every classification also comes with a supporting quote, a reason, and a confidence level. This makes every signal traceable back to the source text.

---

### Component 5 — Python Scoring Engine

This is where the system breaks cleanly with the approach that caused problems in Section 5A.

Instead of asking the language model to produce the final rankings, the system hands the signals to Python and lets pure arithmetic take over. The formula is simple:

```
AI Score  = sum of (signal weight × signal level)
            where none=0.0, partial=0.5, full=1.0
            max AI Score = 2.5

Fin Score = sum of (rank score × metric weight)
            max Fin Score ≈ 2.0

Total     = AI Score + Fin Score   (max = 4.5)
```

The signal weights differ by investor strategy. A conservative investor's weight profile puts heavy emphasis on governance and proven products; a growth investor's profile rewards infrastructure moats and novel research. Same signals, different rankings — because the investors care about different things.

The final recommendation thresholds are fixed:

| Total Score | Recommendation |
|-------------|---------------|
| ≥ 3.25 | **Buy** |
| 2.50 – 3.24 | **Hold** |
| < 2.50 | **Sell** |

Python arithmetic is deterministic. Running the same formula with the same inputs always produces the same answer. This is the property that makes the system trustworthy across runs.

---

### Component 6 — Explanation Layer

Rankings without context aren't useful. Once Python has computed the final scores, the system asks a language model to write a plain-English explanation of why the rankings came out the way they did.

The important design constraint here is that the model is given the pre-computed table as a *fact* — not as something it should recalculate. Its only job is to translate numbers into narrative. It is explicitly instructed not to change scores or rankings, and not to introduce information that isn't in the pre-computed results. This separation guarantees that the explanation never contradicts the numbers.

---

### Component 7 — Evaluation Layer

The final component asks an independent question: how good is this output, really?

A separate, more capable language model — `gpt-4o`, used only here — acts as a judge and scores three things:

| What's being evaluated | The question it answers | Result |
|------------------------|------------------------|--------|
| Signal Groundedness | Is every signal label backed by actual text from the summary? | 5/5 |
| Explanation Consistency | Does the explanation match the pre-computed scores exactly? | 5/5 |
| Explanation Relevance | Does the explanation actually answer the investment question? | 5/5 |

The RAG pipeline is also evaluated separately, including a test where the system is asked about a company that isn't in the dataset. It correctly refused to answer — scoring 5/5 for relevance on that question.

---

## The Three Design Decisions That Define the System

### Why map-reduce, not other summarisation approaches?

Three approaches were considered:

| Approach | How it works | Verdict |
|----------|-------------|---------|
| **Map-Reduce** ✅ | Summarise chunks in parallel, then combine | Fast; all content preserved |
| Refine | Summarise chunk 1, then revise with chunk 2, then chunk 3... | Sequential — significantly slower |
| LLMLingua | Use a compression model to shrink text before sending to LLM | Lossy — drops financial figures, dates, and ticker names |

Map-reduce was the only approach that was both fast and lossless.

### Why Python scoring instead of LLM scoring?

Section 5A (the original approach) asked a language model to produce the entire ranked table, including numeric scores. On long outputs, tiny floating-point variations accumulated across tokens — enough to flip a score from 3.04 to 2.90, or swap two companies' positions between runs.

Section 5B (the deterministic approach) confines the model to one short classification task. All arithmetic moves to Python. The shorter the model's output, the less room for drift. The longer Python handles the math, the more reproducible the results.

### Why cache at every expensive step?

Every computation that takes significant time or API calls writes its result to disk on first run. Three layers of cache exist:

| What's cached | What it avoids |
|--------------|----------------|
| PDF summaries | All map-reduce LLM calls |
| AI signal features | Signal extraction LLM call per company |
| Strategy explanations | Explanation LLM calls per investor strategy |

On every subsequent run, the system loads cached files and skips the heavy lifting entirely. This makes iteration fast and eliminates redundant API costs.

---

## Technology Choices

| Purpose | Tool | Why this choice |
|---------|------|----------------|
| Language model (pipeline) | `gpt-4o-mini` | Cost-efficient; sufficient for classification and summarisation |
| Language model (judge) | `gpt-4o` | More capable; used only for evaluation to ensure quality |
| LLM orchestration | LangChain | Built-in map-reduce chains, async support, prompt templates |
| Vector database | ChromaDB | In-memory, no server required, fast for the document set |
| Financial data | yfinance | Free, reliable, covers all 5 companies |
| Scoring & analysis | Python (pandas, numpy) | Deterministic, auditable, no LLM dependency |
| Visualisation | matplotlib | Standard; sufficient for price trend and bar charts |

---

## What the System Does Not Do

Being clear about limitations matters as much as describing capabilities:

- **Not real-time** — AI signal analysis is based on fixed PDF documents; it goes stale as companies announce new products
- **Not financial advice** — rankings are outputs of a scoring model, not professional investment research
- **Not scalable as-is** — the vector database runs in memory; adding more companies requires persisting it to disk
- **Proxy API dependency** — built and tested against the Great Learning OpenAI-compatible proxy; behaviour on the official OpenAI API may differ slightly
