# Low-Level Design — DualLens-AI

> This document is the implementation companion to [HLD.md](HLD.md). Where the HLD explains *what* each component does and *why* it exists, this document explains *how* it was built — the exact data structures, formulas, prompts, and algorithms that make each piece work. Decision points and tradeoffs are called out inline, where they influenced the implementation.

---

## Module Map

```
DualLens-AI
├── Component 1  — Financial Analysis
│   ├── Data fetch (yfinance)
│   └── Rank-based scoring formula
├── Component 2  — RAG Pipeline
│   ├── PDF loading & chunking
│   ├── ChromaDB embedding & retrieval
│   └── RAG Q&A chain
├── Component 3  — Map-Reduce Summariser
│   ├── Map prompt (per chunk)
│   ├── Reduce prompt (per company)
│   └── Disk cache
├── Component 4  — LLM Signal Extractor
│   ├── Signal schema & rubrics
│   ├── Extraction prompt
│   └── Validation & normalisation
├── Component 5  — Python Scoring Engine
│   ├── AI scoring formula & weights by strategy
│   ├── Financial scoring formula
│   ├── Final ranking & recommendation thresholds
│   └── Weight sensitivity analysis
├── Component 6  — Explanation Layer
│   └── Constrained explanation prompt
└── Component 7  — Evaluation Layer
    ├── Signal Groundedness evaluator
    ├── Explanation Consistency evaluator
    └── Explanation Relevance evaluator
```

---

## Component 1 — Financial Analysis

The financial side of the system is the simplest piece, and deliberately so. Its job is to produce a stable, objective baseline — numbers that don't change based on how a prompt is worded.

The system pulls two things from Yahoo Finance for each company: three years of daily closing prices (used to draw the price trend chart) and five snapshot metrics used for scoring.

```python
companies = ["GOOGL", "MSFT", "IBM", "NVDA", "AMZN"]

ticker = yf.Ticker(symbol)
hist   = ticker.history(period="3y")   # DataFrame: Date × OHLCV

metrics = {
    "Market Cap":     info.get("marketCap"),
    "P/E Ratio":      info.get("trailingPE"),
    "Dividend Yield": info.get("dividendYield", 0),
    "Beta":           info.get("beta"),
    "Total Revenue":  info.get("totalRevenue"),
}
```

Once the metrics are fetched, the scoring logic ranks all five companies on each metric. Each rank position carries a base score. Rank 1 earns 0.40 points; rank 5 earns 0.08. That base score is then multiplied by a strategy-specific weight.

```python
base_scores = [0.40, 0.32, 0.24, 0.16, 0.08]   # rank 1 → rank 5

strategy_weights = {
    "conservative": {
        "Market Cap": 0.8, "Total Revenue": 0.8,
        "P/E Ratio":  1.2, "Beta": 1.2, "Dividend Yield": 1.2
    },
    "balanced": {all metrics: 1.0},
    "growth": {
        "Market Cap": 1.2, "Total Revenue": 1.2,
        "P/E Ratio":  0.8, "Beta": 0.8, "Dividend Yield": 0.8
    },
}
```

For P/E Ratio and Beta, a lower value is better — so the company with the lowest P/E earns rank 1. For all other metrics, higher is better. No weighted score can exceed 0.40, which caps the maximum financial contribution per metric and prevents any single metric from dominating.

> **Decision: Rank-based scoring vs. absolute scoring**
>
> The alternative was to normalise each metric to a 0–1 range and score companies in absolute terms. The problem: NVIDIA's P/E ratio is roughly 10× higher than IBM's. In an absolute scoring scheme, IBM would score near-perfect on P/E while NVIDIA scores near-zero — an accurate reflection of value, but one that dominates the final score unfairly. Rank-based scoring is outlier-resistant: it only asks "who is relatively better?", not "by exactly how much?"
>
> The tradeoff is that rank-based scoring loses magnitude information. A company that is slightly ahead earns the same rank as one that is dramatically ahead. For this dataset of five comparable tech giants, that loss is acceptable. It would matter more in a larger, more heterogeneous universe.

> **Decision: Five snapshot metrics only — no forward-looking indicators**
>
> Revenue growth rate, forward P/E, and free cash flow were considered but excluded. Including them would have made the system more complete but also harder to calibrate weights for — and each additional metric adds a new surface for data quality issues from yfinance. The current five metrics are all well-supported and rarely null. The limitation is acknowledged: a company with flat revenue but accelerating growth (e.g., a turnaround story) will not score well here.

> **Decision: Cap per-metric score at 0.40**
>
> Without the cap, a conservative investor's 1.2× weight on Beta could push a single metric above the contribution ceiling, effectively making Beta the deciding factor in every conservative ranking. The cap ensures no single metric — regardless of strategy weighting — can contribute more than the top-ranked position without weighting.

The final financial score for each company is the sum across all five metrics, with a theoretical maximum of approximately 2.0 points.

---

## Component 2 — RAG Pipeline

The RAG pipeline is where the qualitative half of the analysis begins. Its job is to make the company PDF documents searchable without loading everything into the model at once.

### Breaking the Documents Into Chunks

The first step is to split the PDFs into small, overlapping text segments. Overlapping is important — it ensures that context at the boundary of one chunk isn't completely severed from its neighbours.

```python
loader  = PyPDFDirectoryLoader(path="data/Companies-AI-Initiatives/")
docs    = loader.load()   # List[Document]; metadata: source, page

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name = "cl100k_base",   # matches text-embedding-ada-002 tokenizer
    chunk_size    = 500,             # tokens per chunk
    chunk_overlap = 50,
)
chunks = splitter.split_documents(docs)   # ~109 chunks total
```

Each chunk is then enriched with two metadata tags — `company` (the ticker symbol) and `source` (the PDF filename). These tags become critical at retrieval time.

> **Decision: chunk_size=500, overlap=50**
>
> Smaller chunks (e.g., 200 tokens) would give more precise retrieval — the top-5 results would stay tightly focused on the query. But each chunk would contain too little context for the model to synthesise a useful answer. Larger chunks (e.g., 1,000 tokens) would be more self-contained but would introduce noise into each retrieval result, since the relevant sentence would be buried among unrelated content.
>
> 500 tokens struck a balance: each chunk covers roughly one topic from the PDF, and five chunks (k=5) provide enough breadth for multi-faceted questions. The 50-token overlap prevents key sentences from being split between chunks with no overlap on either side.

> **Decision: `cl100k_base` tokenizer to match `text-embedding-ada-002`**
>
> Using a different tokenizer for chunking than the one used by the embedding model would cause chunk_size=500 to mean different things at chunking time vs. embedding time. A chunk declared as 500 tokens by the splitter might actually contain 620 tokens by the embedding model's count — silently truncating content. Using the same tokenizer at both stages guarantees the declared size is the actual embedded size.

### Building the Searchable Index

Each chunk is converted into a vector — a mathematical representation that captures the semantic meaning of the text — and stored in ChromaDB, an in-memory vector database.

```python
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore     = Chroma.from_documents(
    documents       = chunks,
    embedding       = embedding_model,
    collection_name = "AI_Initiatives",
)
```

> **Decision: ChromaDB in-memory vs. persisted to disk**
>
> ChromaDB supports both in-memory and disk-persisted modes. Persisting to disk would allow the vector store to survive kernel restarts and be reused across sessions. The in-memory mode was chosen because the dataset is small enough (~109 chunks) to re-embed quickly, and keeping everything in memory simplifies the setup — no database files to manage or paths to configure.
>
> The tradeoff becomes a real limitation if the system were scaled to hundreds of companies. At that point, re-embedding on every run would become prohibitively slow, and disk persistence would be required.

### Retrieving Only What's Relevant

When a question comes in, the system identifies which company it refers to and filters the search to only that company's chunks. This company-aware filtering is what prevents Microsoft's content from surfacing when you ask about Amazon.

```python
def extract_companies(query: str) -> list[str]:
    # Keyword match: "Google" → "GOOGL", "Microsoft" → "MSFT", etc.
    # Returns [] if no company is detected — falls back to unfiltered search

def build_retriever(query: str):
    detected = extract_companies(query)
    if len(detected) == 1:
        filter = {"company": detected[0]}         # exact match
    elif len(detected) > 1:
        filter = {"company": {"$in": detected}}   # multi-company query
    else:
        filter = None                             # global fallback
    return vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": filter}
    )
```

> **Decision: Keyword-based company detection vs. LLM-based entity extraction**
>
> Using a language model to extract company names from a query would handle paraphrases and ambiguous references more gracefully ("the GPU giant" → NVIDIA). But it adds an extra model call for every RAG question, increasing latency and cost.
>
> Keyword matching is fast, free, and deterministic. The tradeoff is brittleness: a query phrased as "the company that makes Azure" would not match "MSFT" through keyword lookup. For the test questions in this dataset — which all use company names or ticker symbols explicitly — this was not a problem in practice.

### Answering Questions with Grounded Context

The RAG Q&A chain is intentionally constrained. The system prompt tells the model to answer *only* from the retrieved chunks — not from its training data. If the chunks don't contain an answer, the model should say so.

```python
qna_system_message = """
You are an AI assistant specialized in analyzing company AI initiatives.
Answer ONLY from the provided context.
If the context does not contain enough information, respond:
"I don't have enough information to answer this question."
"""

def RAG(question: str) -> str:
    context  = build_ai_context_for_user_question(question)
    messages = [SystemMessage(qna_system_message),
                HumanMessage(user_template.format(context=context, question=question))]
    return llm.invoke(messages).content
```

> **Decision: Constrained answers vs. allowing the model to use its training knowledge**
>
> A language model trained on public data already "knows" quite a lot about Microsoft's Azure and NVIDIA's CUDA. Allowing it to supplement retrieved context with training knowledge would produce richer answers — but those answers would be unverifiable. The Groundedness evaluator cannot check claims that come from training data rather than the retrieved chunks.
>
> The constraint — answer only from context — makes every claim traceable and evaluable. The cost is that the system will sometimes say "I don't have enough information" when a knowledgeable human analyst would know the answer. That is the right tradeoff for a system that claims to be grounded.

---

## Component 3 — Map-Reduce Summariser

While RAG handles question-answering, the scoring engine needs something different: a complete, coherent summary of each company — one that fits in a single model call and doesn't lose anything important.

That's the map-reduce summariser's job. It processes every chunk, summarises them independently in parallel (the map step), then combines those summaries into one final paragraph per company (the reduce step).

> **Decision: Map-Reduce vs. Refine vs. LLMLingua**
>
> Three approaches were evaluated:
>
> | Approach | How it works | Key tradeoff |
> |----------|-------------|--------------|
> | **Map-Reduce** ✅ | Summarise each chunk independently (parallel), then combine | Fast; all content preserved; each chunk summarised without awareness of others |
> | Refine | Summarise chunk 1, then revise with chunk 2, then chunk 3... | Each summary is contextually aware of prior chunks, but sequential — far slower for 109 chunks |
> | LLMLingua | Compress text using a smaller model (BERT-class) before sending to the LLM | Faster and cheaper; but silently drops structured data — financial figures, dates, and ticker names were lost in testing |
>
> Refine's sequential nature would have made the summarisation step take hours rather than minutes. LLMLingua's lossy compression was the more dangerous failure: it dropped exactly the kind of specific evidence (revenue figures, product names) that the signal extractor needs. Map-reduce was the only approach that was both fast and information-preserving.

### The Map Prompt

The map prompt runs against every chunk independently. It explicitly calls out infrastructure position because early testing showed that generic summarisation prompts would silently drop NVIDIA's CUDA dominance — one of the most investment-relevant signals in the entire dataset.

```
Summarize the following excerpt from {company}'s AI initiatives document.
Focus on:
- Specific AI products, platforms, or research projects
- Enterprise or consumer deployment status
- INFRASTRUCTURE POSITION (critical):
    Is the company's hardware/software the foundational compute layer for AI industry-wide?
    Proprietary chips, CUDA ecosystem, platforms others build AI on top of
    Whether described as "backbone", "standard", or "dominant" in AI compute
    NVDA: explicitly state whether NVIDIA GPUs/CUDA serve as foundational AI compute infrastructure
```

> **Decision: Signal-aware map prompt vs. generic summarisation**
>
> A generic map prompt ("summarise this excerpt") would produce shorter, cleaner summaries — but it would allow the model to decide which details are worth keeping. In practice, the model treated NVIDIA's infrastructure dominance as background context rather than a key fact, omitting it from summaries. When downstream signal extraction then read those summaries, `infrastructure_moat` was classified as `none` for all companies — including NVIDIA.
>
> Making the map prompt signal-aware solves this by explicitly instructing the model what to preserve. The tradeoff is that the prompt becomes less general: it is tuned to this specific set of eight signals. Extending the system to new signals would require updating the map prompt accordingly.

### The Reduce Prompt

The reduce prompt combines the chunk summaries into a single paragraph per company. Its most important rule: do not introduce information that wasn't in the summaries.

```
Combine the following summaries of {company}'s AI initiatives into one coherent paragraph.
Rules:
- ALWAYS include infrastructure position if the company's hardware/software IS the
  foundational compute layer for AI industry-wide
- Preserve specific product names, timelines, and revenue figures
- Do not introduce information not present in the summaries
```

### Caching the Result

Because the map step makes dozens of parallel model calls, the full summarisation process takes significant time and API budget. The result is saved to disk immediately after generation. On every subsequent run, the system loads the cached file and skips summarisation entirely.

```python
cache_path = BASE_DIR / "cache" / "pdf_summary.txt"

def get_or_generate_AI_summary(force_regenerate=False):
    if not force_regenerate and cache_path.exists():
        return cache_path.read_text()
    summary = chain.run(chunks)
    cache_path.write_text(summary)
    return summary
```

> **Decision: Cache to plain text vs. structured format (JSON)**
>
> Storing the summaries as plain text with `**TICKER:**` section headers is simpler to write, read, and inspect manually. The tradeoff is that parsing requires a simple string split rather than a `json.loads()` call. Given that the format is stable and the file is written once by the system itself, the simplicity of plain text outweighed the slightly cleaner parsing of JSON.

---

## Component 4 — LLM Signal Extractor

With compact, per-company summaries available, the system can now extract the eight investment signals. This is the only step in the deterministic pipeline where a language model is asked to make a judgment — and the task is intentionally narrow.

### The Signal Schema

Every signal returns the same structure: a classification level, a supporting quote, a reason for the classification, and a confidence assessment.

```json
{
  "enterprise_ai": {
    "level":      "full",
    "evidence":   "Azure OpenAI Service used by enterprise customers",
    "reason":     "Commercial enterprise AI platform with broad business use",
    "confidence": "high"
  }
}
```

The three levels map directly to numbers for scoring:

| Label | Numeric | Meaning |
|-------|---------|---------|
| `none` | 0.0 | Signal absent from the summary |
| `partial` | 0.5 | Signal present but limited in scope |
| `full` | 1.0 | Signal clearly and fully present |

> **Decision: Three discrete levels vs. a continuous 0–10 scale**
>
> The first version of the extractor asked the model to assign a score between 0 and 10 for each signal. The scores were inconsistent between runs — the same evidence might produce a 7 in one run and a 6.5 in the next, even at temperature=0. This was the root cause of the reproducibility failure in Section 5A.
>
> Three discrete levels constrain the model's decision space dramatically. Choosing between `none`, `partial`, and `full` is a much simpler task than picking a number on a continuous scale. The output is also much shorter — a single word per signal rather than a floating-point number with reasoning — which gives less room for token-level variance to accumulate. The tradeoff is reduced granularity: two companies with meaningfully different evidence strengths both get `full` if they both clearly exhibit the signal.

### The Rubrics

Rubrics define exactly what counts as `none`, `partial`, or `full` for each signal. Without rubrics, two runs of the same prompt can produce different levels for the same evidence.

| Signal | `none` | `partial` | `full` |
|--------|--------|-----------|--------|
| `enterprise_ai` | No enterprise product | Pilot or limited enterprise use | Commercial enterprise AI platform |
| `consumer_ai` | No consumer product | Beta or limited consumer release | Widely deployed consumer AI product |
| `hardware_ai` | No AI hardware | Custom AI chips with limited adoption | Proprietary AI silicon in production |
| `public_release` | No public product | Research preview or beta | Fully deployed, publicly available |
| `core_business_alignment` | AI is peripheral | AI enhances some business lines | AI is central to primary revenue model |
| `novel_research` | No disclosed research | Research mentioned without specifics | Named research project with described impact |
| `risk_mitigation_present` | No risk management | Risks acknowledged without plan | Active governance program described |
| `infrastructure_moat` | No foundational role | Cloud AI infra with moderate adoption | De facto standard for AI compute industry-wide (e.g. CUDA) |

> **Decision: Explicit rubrics in the prompt vs. relying on model judgment**
>
> Without rubrics, "partial" for `enterprise_ai` was interpreted inconsistently — sometimes meaning "they have an enterprise product that isn't widely adopted," sometimes meaning "they have enterprise features inside a general-purpose tool." The rubric anchors the label to a specific definition that the evaluator can later check. This is what allows the Groundedness evaluator to ask: "Is this `full` label actually supported by the summary?"
>
> The tradeoff is that rubrics add significant prompt length, which increases cost per extraction call. They also require human judgment to write well — poorly worded rubrics can create ambiguous boundaries that produce inconsistent labels of a different kind.

### The Extraction Prompt

The extraction prompt presents all five companies' summaries in one call and asks for a single JSON object covering all companies. This keeps the result self-consistent — the model can calibrate what "full" means for NVIDIA's infrastructure position against what "partial" means for Amazon's in the same context.

```
System:
  You are a financial analyst. Extract AI initiative signals for each company
  from the provided summary. For each signal return:
    - level: "none" | "partial" | "full" (use rubrics below)
    - evidence: direct quote or paraphrase from the summary
    - reason: one sentence explaining the classification
    - confidence: "high" | "medium" | "low"

  [Per-signal rubrics as defined above]

User:
  AI Summary:
  {summary}

  Return valid JSON matching exactly this schema:
  {
    "MSFT": { "enterprise_ai": {...}, "consumer_ai": {...}, ... },
    "GOOGL": { ... },
    "AMZN":  { ... },
    "NVDA":  { ... },
    "IBM":   { ... }
  }
```

> **Decision: All five companies in one prompt vs. one prompt per company**
>
> Running one prompt per company would produce shorter, simpler prompts — easier for the model to process and less likely to hit context limits. But it would lose cross-company calibration: the model classifying NVDA's `infrastructure_moat` as `full` in isolation might classify AMZN's as `partial` in isolation, but if both summaries are in the same context, the model can more consistently calibrate that AMZN's cloud AI infrastructure is genuinely a step below NVDA's CUDA ecosystem.
>
> The five-company prompt is longer but produces more internally consistent labels. The tradeoff is a larger prompt that costs more per call and is more sensitive to model context limits.

### Validation and Normalisation

Model outputs aren't always perfectly structured. The validation layer handles the common failure modes: missing fields, invalid level values, and signals returned as plain strings instead of objects.

```python
_EXPECTED_SIGNALS = [
    "enterprise_ai", "consumer_ai", "hardware_ai", "public_release",
    "core_business_alignment", "novel_research",
    "risk_mitigation_present", "infrastructure_moat",
]

def _normalise_signal(val) -> dict:
    # Accepts: dict with level/evidence/reason/confidence
    #          or plain string "none"/"partial"/"full"
    # Returns: standardised dict; fills missing fields with defaults
    # Coerces any invalid level value to "none"

def _parse_features_robust(raw_json: str) -> dict:
    # Tries json.loads(); falls back to regex extraction on failure
    # Validates all 5 companies are present
    # Validates all 8 signals are present per company
    # Calls _normalise_signal() on each value
```

> **Decision: Silent coercion vs. hard failure on invalid output**
>
> When the model returns an invalid level value (e.g., `"yes"` instead of `"full"`), the normaliser coerces it to `"none"` rather than raising an exception. This keeps the pipeline running even when the model produces slightly malformed output. The tradeoff is that a silent `"none"` can undercount a company's signal strength without any visible warning. In production, this would warrant an alert and a human review step.

The validated features are saved to `cache/ai_features.json` and loaded from there on subsequent runs.

---

## Component 5 — Python Scoring Engine

This is the heart of the deterministic approach. Everything from here on is pure Python — no model calls, no variability.

> **Decision: Python scoring (Section 5B) vs. LLM scoring (Section 5A)**
>
> Section 5A asked a language model to produce the entire ranked table including numeric scores. On long outputs — three concurrent calls each producing ~3,000 tokens — tiny floating-point variations accumulated enough to flip scores across the Buy/Hold boundary between runs, even at temperature=0. Rankings were non-reproducible.
>
> Section 5B confines the model to the narrowest possible task — three-way signal classification — and hands all arithmetic to Python. Python's floating-point arithmetic is deterministic for the same inputs on the same platform. The same formula always produces the same answer.
>
> The cost of this decision is that the scoring model is only as good as its 8 signals and their weights. A company that is doing something genuinely important in AI that doesn't map to any of the eight signals will not be captured. Section 5A's unconstrained LLM could potentially surface such nuance — but unreliably.

### AI Scoring

Each company's AI score is the weighted sum of its signal levels. The weights differ by strategy because different investors care about different signals.

```python
SIGNAL_LEVELS = {"none": 0.0, "partial": 0.5, "full": 1.0}

def compute_ai_score(signals: dict, strategy: str) -> float:
    weights = AI_SIGNAL_WEIGHTS_BY_STRATEGY[strategy]
    return round(
        sum(weights[k] * SIGNAL_LEVELS[signals[k]["level"]]
            for k in weights if k in signals),
        2
    )
```

The weight profiles reflect each investor's priorities:

> **Decision: Why `hardware_ai` and `infrastructure_moat` were added**
>
> The initial six signals (`enterprise_ai`, `consumer_ai`, `public_release`, `core_business_alignment`, `novel_research`, `risk_mitigation_present`) all measure how a company deploys AI products to customers. They systematically underscore NVIDIA, which has no consumer or enterprise AI products — it builds the infrastructure every other company's AI runs on. Without these two signals, NVIDIA would rank last under every strategy: a result that misrepresents the only company whose hardware and software the entire AI industry depends on.
>
> `hardware_ai` captures silicon ownership — NVIDIA is the only company scoring `full`. `infrastructure_moat` captures foundational compute dependency — NVIDIA's CUDA ecosystem is the de facto standard; MSFT/GOOGL/AMZN score `partial` for their cloud platforms.
>
> Under Conservative weights both signals contribute only 0.40 pts — NVIDIA's zero enterprise/consumer signals still dominate and it ranks Sell. Under Growth weights they contribute 1.10 pts, which is enough to flip NVIDIA to Hold. This Growth flip is the most investment-relevant finding in the project: NVIDIA is a viable investment only for growth-oriented investors, and only because of infrastructure dominance.

```python
AI_SIGNAL_WEIGHTS_BY_STRATEGY = {
    "conservative": {
        "core_business_alignment": 0.50,
        "enterprise_ai":           0.40,
        "public_release":          0.50,   # proven execution matters most
        "hardware_ai":             0.30,
        "consumer_ai":             0.20,
        "novel_research":          0.10,
        "risk_mitigation_present": 0.50,   # governance is critical for conservative
        "infrastructure_moat":     0.10,
    },
    "balanced": {
        "core_business_alignment": 0.50,
        "enterprise_ai":           0.50,
        "public_release":          0.40,
        "hardware_ai":             0.40,
        "consumer_ai":             0.30,
        "novel_research":          0.20,
        "risk_mitigation_present": 0.20,
        "infrastructure_moat":     0.30,
    },
    "growth": {
        "core_business_alignment": 0.50,
        "enterprise_ai":           0.50,
        "public_release":          0.30,
        "hardware_ai":             0.60,   # hardware dominance is the key moat
        "consumer_ai":             0.30,
        "novel_research":          0.30,
        "risk_mitigation_present": 0.10,
        "infrastructure_moat":     0.50,   # infrastructure moat drives growth thesis
    },
}
```

The maximum AI score is 2.5 (all eight signals at `full`).

> **Decision: Manually defined weights vs. learned weights**
>
> The weights were hand-tuned to reflect investment logic for each strategy profile rather than derived from data. Learned weights (from historical returns, for example) would be more empirically grounded — but they would require a historical dataset of AI signal scores and subsequent stock performance that doesn't exist for this small set of companies.
>
> Manual weights are transparent and auditable. An investor can look at the conservative profile and immediately see that governance (`risk_mitigation_present: 0.50`) is weighted equally to proven execution (`public_release: 0.50`). That's a meaningful investment thesis that can be debated and adjusted. Learned weights would be harder to interpret and potentially overfit to the historical window used to derive them.

### Final Ranking and Recommendation

The total score is AI score plus financial score. The ranking function sorts companies by total score, breaking ties by AI score.

```python
def rank_companies_det(ai_scores: dict, fin_totals: dict) -> list[dict]:
    rows = []
    for ticker in companies:
        ai    = ai_scores[ticker]        # 0.0 – 2.5
        fin   = fin_totals[ticker]       # 0.0 – ~2.0
        total = round(ai + fin, 2)       # 0.0 – 4.5
        rows.append({
            "ticker":    ticker,
            "ai_score":  ai,
            "fin_score": fin,
            "total":     total,
            "rec":       get_rec(total),
        })
    return sorted(rows, key=lambda x: (-x["total"], -x["ai_score"]))

def get_rec(total: float) -> str:
    if total >= 3.25: return "Buy"
    if total >= 2.50: return "Hold"
    return "Sell"
```

> **Decision: Fixed recommendation thresholds vs. relative thresholds**
>
> Fixed thresholds (Buy ≥ 3.25, Hold ≥ 2.50) are simple and transparent — but they create hard boundaries. A company scoring 3.24 is called "Hold" while a company scoring 3.25 is called "Buy", even though the underlying difference is negligible. Relative thresholds (e.g., top 2 companies = Buy, bottom 2 = Sell) would avoid this cliff effect but would require a Buy recommendation regardless of how weak the top companies actually are.
>
> Fixed thresholds were chosen because they give the system an absolute opinion — "this company is genuinely investable" — rather than a relative one. The cliff effect is a known limitation.

### Weight Sensitivity Analysis

Before trusting the rankings, the system checks whether they would survive small changes to the weight configuration. It shifts each signal weight by ±20%, one at a time, and checks whether the rank order changes.

```python
def sensitivity_analysis(features, df_yahoo, strategy="balanced", delta=0.20):
    baseline_order = [r["ticker"] for r in rank_companies_det(...)]

    results = []
    for sig in _EXPECTED_SIGNALS:
        for direction in [+delta, -delta]:
            tweaked_weights = {**AI_SIGNAL_WEIGHTS_BY_STRATEGY[strategy]}
            tweaked_weights[sig] = max(0, tweaked_weights[sig] * (1 + direction))

            tweaked_ai    = {t: compute_ai_score_with_weights(features[t], tweaked_weights)
                             for t in companies}
            tweaked_order = [r["ticker"] for r in rank_companies_det(tweaked_ai, fin_totals)]

            results.append({
                "signal":    sig,
                "shift":     f"{'+' if direction > 0 else ''}{int(direction*100)}%",
                "stable":    tweaked_order == baseline_order,
                "new_order": " > ".join(tweaked_order),
            })
    return results
```

> **Decision: One-at-a-time perturbation vs. combinatorial sensitivity analysis**
>
> Shifting one weight at a time across ±20% produces 16 test cases (8 signals × 2 directions) and is easy to compute and read. A full combinatorial analysis — every possible combination of weight shifts — would produce 2^16 = 65,536 test cases and would be much harder to summarise into a useful stability verdict.
>
> The one-at-a-time approach answers the practical question: "Is any single signal driving the ranking?" If no individual weight shift flips the order, the ranking is robust to small calibration errors in any one weight. It does not catch cases where two weights are simultaneously miscalibrated in complementary directions — but that's a second-order concern for a five-company system.

---

## Component 6 — Explanation Layer

The explanation layer is where the language model re-enters — but in a carefully constrained role. It receives the pre-computed ranking table and signal evidence as fixed inputs and is instructed to write a narrative that explains them, not re-derive them.

```python
DET_EXPLANATION_SYSTEM = """
You are a financial analyst writing an investment report.
Use ONLY the pre-computed ranking table and signal evidence provided.
Do NOT change scores, rankings, or recommendations.
Do NOT introduce information not present in the inputs.
Structure: 1 paragraph overview + per-company bullet points.
"""

DET_EXPLANATION_USER = """
Strategy: {strategy}
Pre-computed Rankings:
{ranking_table}
Signal Evidence:
{signal_evidence}
AI Summary:
{ai_summary}
"""
```

> **Decision: Constrained explanation vs. free narration**
>
> The alternative was to let the model narrate the investment case freely, giving it access to the signal evidence but not the pre-computed scores — allowing it to form its own view. The risk is that a freely narrating model might rank companies differently from the Python engine, producing an explanation that contradicts the ranking table.
>
> The constrained approach keeps scores and narrative in lockstep by construction. The Consistency evaluator scored this component 5/5, meaning the prompt constraints successfully kept the narrative aligned with the pre-computed scores. Perfect consistency from a language model is not guaranteed — the constraint works here because the pre-computed table is provided as an explicit fact and the model is instructed not to deviate from it.

> **Decision: LLM-written prose vs. a structured template**
>
> A template-generated explanation (e.g., `{ticker} earns a {rec} with a total score of {total}...`) would be perfectly accurate — it reads directly from the Python output. But it would be robotic and fail to explain *why* the ranking came out as it did.
>
> The LLM explanation scored 5/5 on both consistency and relevance — it accurately reflected the pre-computed scores and genuinely answered the investment question in a way a human investor would find useful. A template would guarantee consistency but produce robotic output with no explanatory value.

The three strategy explanations are saved to `cache/strategy_explanations.json` and loaded from cache on subsequent runs.

---

## Component 7 — Evaluation Layer

The evaluation layer is the quality gate. Three separate judge prompts — all using `gpt-4o` — score different aspects of the system's output. Each follows the same chain-of-thought structure: Steps 1–4 analyse a specific quality dimension, and Step 5 returns a score from 1 to 5.

> **Decision: `gpt-4o` as judge vs. `gpt-4o-mini`**
>
> The pipeline uses `gpt-4o-mini` for all production steps — it is cheaper and fast enough for classification and summarisation. The evaluation layer uses `gpt-4o` because quality judgments are more nuanced: detecting whether an explanation "contradicts" a pre-computed score requires the model to compare two texts at a level of precision that smaller models handle inconsistently.
>
> The cost difference is acceptable because evaluation runs only once, not on every re-run of the pipeline. Spending more on the judge than on the pipeline itself is a reasonable tradeoff when the judge's verdict is used to validate the pipeline's trustworthiness.

> **Decision: Chain-of-thought rubric (Steps 1–4 before scoring) vs. direct score request**
>
> Asking the model to score directly ("Rate the groundedness of these signals from 1 to 5") produces inconsistent results — the model tends to anchor on first impressions and give high scores to plausible-sounding outputs regardless of whether they are actually grounded.
>
> The chain-of-thought rubric forces the model to reason through specific sub-dimensions before committing to a score. Step 1 through Step 4 act as a structured audit; Step 5 synthesises them. This produces more calibrated and more consistent scores at the cost of longer, more expensive judge outputs.

> **Decision: Three separate evaluators vs. a single composite score**
>
> A single evaluator asking "how good is this output overall?" would produce a single number that blends groundedness, consistency, and relevance together. If that number is low, you don't know which dimension failed.
>
> Three separate evaluators give diagnostic precision: each dimension can be assessed independently. In this run all three scored 5/5 — Signal Groundedness, Explanation Consistency, and Explanation Relevance — confirming that signal labels are traceable, the narrative matches the pre-computed scores, and the explanation answers the investment question. If any dimension had scored low, the separation would identify exactly where to improve rather than just "overall quality needs work."

### Signal Groundedness Evaluator

Checks that every signal classification is traceable to text in the AI summary. A label of `full` with no supporting quote should score low; a label with a direct paraphrase should score high.

```
Given:
  - AI summary text for all 5 companies
  - Extracted signals with level + evidence for all 5 companies

Step 1 — Evidence Quality:          Is each "full"/"partial" label backed by a
           direct quote or clear paraphrase from the summary?
Step 2 — Partial/None Accuracy:     Are "none" labels correct? Is the summary
           genuinely silent on that signal?
Step 3 — Confidence Calibration:    Does the confidence level match the strength
           of evidence cited?
Step 4 — Cross-Company Consistency: Are similar situations labelled the same
           way across companies?
Step 5 — Final Score: 1–5
```

### Explanation Consistency Evaluator

Checks that the LLM-written narrative doesn't contradict or embellish the pre-computed scores. The key failure mode it catches is a narrative that cites a score that doesn't match the Python output.

```
Given:
  - Pre-computed ranking table (Python output)
  - LLM-written explanation narrative

Step 1 — Score Accuracy:      Does the narrative cite the correct numeric scores?
Step 2 — Signal Attribution:  Does the narrative attribute rankings to the right signals?
Step 3 — Reasoning Coherence: Is the comparative reasoning logically sound?
Step 4 — No Hallucination:    Does the narrative introduce claims not in the table?
Step 5 — Final Score: 1–5
```

### Explanation Relevance Evaluator

Checks that the explanation actually answers the original investment question — not a tangential version of it.

```
Given:
  - Investment question: "Which company is the best investment?"
  - LLM-written explanation narrative

Step 1 — Core Intent:    Does the explanation identify which company to invest in?
Step 2 — Relevance:      Does it stay on-topic (AI + financial signals)?
Step 3 — Completeness:   Does it address all 5 companies?
Step 4 — Actionability:  Does it give Buy/Hold/Sell reasoning for each?
Step 5 — Final Score: 1–5
```

---

## Scoring Summary

| Score Component | Formula | Max |
|----------------|---------|-----|
| AI Score | `Σ (signal_weight × SIGNAL_LEVELS[level])` | 2.5 |
| Financial Score | `Σ (rank_score × strategy_weight)` | ~2.0 |
| Total Score | `AI Score + Financial Score` | 4.5 |

| Recommendation | Threshold |
|---------------|-----------|
| **Buy** | Total ≥ 3.25 |
| **Hold** | 2.50 ≤ Total < 3.25 |
| **Sell** | Total < 2.50 |

---

## Cache File Reference

| File | What it stores | When it's written |
|------|---------------|-------------------|
| `cache/pdf_summary.txt` | One paragraph per company, map-reduce output | After first summarisation run |
| `cache/ai_features.json` | 5 companies × 8 signals, validated JSON | After first extraction run |
| `cache/strategy_explanations.json` | One explanation string per strategy | After first explanation run |
| `cache/strategy_recommendations.json` | Section 5A LLM-generated ranked tables | After first Section 5A run |
