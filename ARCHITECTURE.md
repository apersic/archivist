# Architecture

Archivist answers questions about a private document corpus, with citations,
and refuses to answer when the corpus does not contain the answer.

This document explains the shape of the system and why each piece is shaped
that way. It is written to be read in order: goals, layers, contracts,
decisions, failure modes, then the path to production scale.

## 1. Goals and non-goals

**Goals**

1. Every answer is traceable to passages a human can open and check.
2. Retrieval quality and answer quality are measured separately and
   continuously, so a change can be shown to help rather than argued to help.
3. Abstention is a first-class outcome, not an error path.
4. Every model and store is replaceable behind a port, so the system can start
   dependency-free on a laptop and end up on hosted models and a vector
   database without rewriting the pipeline.
5. The whole pipeline runs offline and deterministically, so tests assert on
   system behaviour rather than on model mood.

**Non-goals**

* Being a framework. There is no plugin system, no YAML DSL, no agent
  orchestration engine. Those are the parts of a RAG stack that are cheap to
  write and expensive to debug.
* Serving a web-scale corpus. The default index is exact and in-memory; see
  §8 for the point at which that must change and what replaces it.
* Multi-tenancy, auth and quotas. Called out in §9 as deliberately deferred.

## 2. The shape of the system

```
                          ┌──────────────────────────────────────────┐
  corpus/  ──loaders──▶   │  INGEST                                  │
  (md, txt, …)            │  load → normalise → structure-aware      │
                          │  chunk (heading path + token windows)    │
                          └───────────────────┬──────────────────────┘
                                              │ Chunk[]
                          ┌───────────────────▼──────────────────────┐
                          │  INDEX                                   │
                          │  Embedder ─▶ VectorIndex (cosine)        │
                          │  LexicalIndex (BM25)                     │
                          └───────────────────┬──────────────────────┘
                                              │
  question ──▶ ┌───────────────────────────────▼─────────────────────┐
               │  RETRIEVE                                           │
               │  rewrite ─▶ expand ─▶ dense ∥ lexical ─▶ RRF fuse   │
               │  ─▶ neighbour expansion ─▶ rerank ─▶ assemble       │
               └───────────────────────────────┬─────────────────────┘
                                               │ AssembledContext ([S1], [S2]…)
               ┌───────────────────────────────▼─────────────────────┐
               │  GENERATE                                           │
               │  grounded prompt ─▶ LLM ─▶ citation validation      │
               │  ─▶ answer | abstention                             │
               └───────────────────────────────┬─────────────────────┘
                                               │ Answer
               ┌───────────────────────────────▼─────────────────────┐
               │  AGENT (optional, bounded)                          │
               │  tool loop: search_corpus · fetch_chunk ·           │
               │  list_sources · answer    budgets: steps, searches  │
               └───────────────────────────────┬─────────────────────┘
                                               │
               ┌───────────────────────────────▼─────────────────────┐
               │  EVALUATE + OBSERVE                                 │
               │  golden set ─▶ retrieval & answer metrics ─▶        │
               │  baseline comparison (CI gate);  trace spans        │
               └─────────────────────────────────────────────────────┘
```

Data flows downward; nothing flows back up. A layer may only know about the
layers above it through the data contracts in `types.py`.

## 3. Contracts

Two files define the whole seam structure of the codebase.

`src/archivist/types.py` holds the data every layer agrees on: `Document`,
`Chunk`, `Scored`, `AssembledContext`, `Citation`, `Answer`, plus the LLM
message/tool-call shapes. They are frozen dataclasses: a retrieved chunk that
can be mutated three layers later is a debugging problem waiting to happen.

`src/archivist/ports.py` holds the five protocols that make models
replaceable: `Embedder`, `VectorIndex`, `LexicalIndex`, `Reranker`, `LLM`.
Each has a dependency-free default implementation and a real one:

| Port | Default (offline) | Production adapter |
|---|---|---|
| `Embedder` | `HashingEmbedder` — feature-hashed word + char n-grams | `OpenAIEmbedder`, or sentence-transformers |
| `VectorIndex` | `InMemoryVectorIndex` — exact cosine, JSONL persistence | pgvector / Qdrant |
| `LexicalIndex` | `BM25Index` — ~60 lines | OpenSearch, or Postgres FTS |
| `Reranker` | `IdentityReranker` (default) / `LexicalOverlapReranker` | `CrossEncoderReranker` |
| `LLM` | `ExtractiveLLM` — deterministic sentence selection | `OpenAIChatLLM` |

`src/archivist/app.py` is the only module that chooses which implementation to
use. Nothing else imports a provider.

## 4. Decisions, and why

**Chunking is structure-aware, and chunks carry their heading path.**
Chunking decides what "one fact" means, so it is the highest-leverage knob in
the system. Sections are split on Markdown headings first, then windowed to a
token budget with overlap, breaking on paragraph and then sentence boundaries.
Each chunk's indexed text is prefixed with `Title — Section`, which makes a
chunk retrievable by terms that appear only in its heading — the fix for the
classic failure where a table row like `| events_raw | 30 days |` is
unretrievable because the word "retention" is three lines above it.

**Retrieval is hybrid, fused by reciprocal rank.**
Dense retrieval fails predictably on rare exact strings: error codes, config
keys, surnames. BM25 fixes exactly that failure and costs almost nothing.
Fusion is RRF rather than score normalisation because cosine similarities and
BM25 scores are not on a comparable scale, and normalising them adds a tuning
parameter per corpus. RRF adds one constant that is famously insensitive.

**Query transforms run before retrieval, not inside it.**
Follow-up questions ("and what about its retention?") retrieve nothing on
their own, so they are rewritten into standalone questions against the
conversation. A second, keyword-only probe of the same question is fused in,
which buys recall on vocabulary mismatch without widening `k`.

**Reranking is a separate stage, and it is off by default.**
Retrieval optimises recall over a wide candidate set; reranking optimises
precision over a narrow one. Keeping them separate is what lets you retrieve
40 candidates cheaply and spend an expensive model on six. The default is
`identity` because on this corpus the ablation showed the heuristic reranker
*lowering* answer quality (`answer_match` 0.833 → 0.750). The stage exists,
the ablation is one flag (`--reranker lexical-overlap`), and the number
decides. A cross-encoder is expected to win; it has not been measured here, so
it is not the default.

**Generation defends against its own model.**
The generator refuses to call the model when retrieval returned nothing;
parses the `[S*]` markers the model emitted and drops any that do not exist in
the context (a fabricated citation becomes a dropped one); and downgrades an
answer with no valid citations to an abstention. An answer nobody can check is
worse than no answer.

**The agent loop is optional and bounded.**
Plain RAG — one retrieval, one generation — is the default and the baseline to
beat. The loop exists for multi-hop questions and for queries whose useful
search terms only appear after a first lookup. Three budgets keep it an agent
rather than an expensive infinite regress: a step budget, a search budget, and
an explicit terminal `answer` tool. Budgets are enforced by the loop, not by
trusting the model to comply: when the search budget is spent the tool is
withdrawn, and a model that asks anyway gets a refusal as its next
observation. (That hole existed in the first draft; the test
`test_search_budget_is_enforced_even_if_the_model_ignores_it` is why it
doesn't now.)

**Evaluation is part of the system, not a notebook.**
Retrieval and generation fail differently and are measured separately: score
only the final answer and you cannot tell a retrieval miss from a generation
miss, and you will tune prompts to fix an indexing bug. Golden labels are
attached to *document paths*, not chunk ids, so the labels survive a
re-chunking — otherwise every chunking experiment invalidates its own ground
truth.

**The offline LLM is a design tool, not a shortcut.**
`ExtractiveLLM` is a deterministic policy that selects and cites the
best-covering sentences from the context, abstains below a coverage floor, and
drives the agent loop with a simple plan. It makes the pipeline, the loop and
the eval harness runnable with no API key, no cost and no flakiness. The tests
therefore assert on system properties — citations resolve to real chunks,
budgets hold, abstention triggers — which remain true of any model you swap in.

## 5. Metrics

Retrieval, per case, against the documents a case declares relevant:

* `source_recall@k` — did the right documents reach the context window?
* `mrr`, `ndcg@k` — were they near the top?

Generation:

* `groundedness` — fraction of the answer's content terms present in the
  retrieved context. A cheap, deterministic hallucination detector.
* `citation_support` — do the cited passages actually share content with the
  answer?
* `answer_match` — fraction of the facts a correct answer must contain.
* `abstention_accuracy` — did it refuse exactly on the unanswerable cases?

`LLMJudge` implements the same call signature as the groundedness proxy for
when a model's opinion is wanted. Calibrate the proxy against the judge on a
sample before trusting either.

Current measured state on the bundled 5-document corpus and 12-case golden set,
with the offline extractive model (`make eval`):

| strategy | source_recall@5 | mrr | ndcg@5 | groundedness | citation_support | answer_match | abstention_acc | pass_rate |
|---|---|---|---|---|---|---|---|---|
| plain-rag | 1.000 | 1.000 | 0.993 | 1.000 | 1.000 | 0.833 | 0.917 | 0.750 |
| agentic | 1.000 | 0.917 | 0.944 | 1.000 | 1.000 | 0.833 | 0.917 | 0.750 |

Read that honestly: **retrieval is solved on this corpus and the answer layer
is the bottleneck.** Every remaining failure is the extractive stand-in
picking a sentence that is on-topic but not the one asked for, and one
over-answer on a question it should have refused. That is exactly the gap a
real LLM closes, and the harness is what will prove it did.

## 6. Observability

`obs/trace.py` records a span per stage with timings, attributes and errors,
in an OpenTelemetry-shaped structure so it can be exported later without
touching callers. A bad answer is usually bad in one specific stage; the trace
is what makes the failure attributable rather than mysterious. Token usage is
accumulated on the trace, which is also the hook for cost accounting.

```
$ archivist research "Which team owns the Heron transform job?" -v
step 1: search_corpus({'query': '…'})
- agent (12.4ms) question=… max_steps=4
  - agent.step (6.1ms) index=1 tool=search_corpus evidence=6
    - retrieve (5.6ms) …
      - retrieve.dense (1.2ms) queries=2 hits=40
      - retrieve.lexical (2.1ms) queries=2 hits=34
      - retrieve.fuse (0.4ms) lists=4 candidates=27
```

## 7. Failure modes and the stage that owns them

| Symptom | Likely stage | What to look at |
|---|---|---|
| Answer cites the right document but the wrong fact | generation | the selected passages in the trace; raise `rerank_limit` or shrink chunks |
| Right document never retrieved | ingest / index | `source_recall@k`; check the chunk's heading prefix |
| Exact identifier not found | lexical index | is the token being split? `tokenize` keeps `_` inside tokens on purpose |
| Answer is fluent and wrong | generation | `groundedness`; require citations; check for dropped invented markers |
| Follow-up question retrieves nothing | query rewrite | `retrieve.rewrite` span; `needs_rewrite` heuristics |
| Agent loops | agent | step/search budgets; the refusal observation in the trace |
| Metrics moved but nobody knows why | config | `Settings.fingerprint()` is recorded in every eval report |

## 8. Scaling path

The default index is exact cosine search in Python dictionaries. That is the
right starting point: it removes ANN recall loss from the variables while
chunking and prompts are still moving, and a few thousand chunks do not need a
database. It stops being right at roughly 10^5 chunks or when the index must
outlive a process.

The replacement is a `VectorIndex` implementation over pgvector or Qdrant plus
a real lexical engine. Three things make that migration safe rather than
exciting: the port is four methods, the golden set measures `source_recall@k`
before and after, and the baselines make a recall loss from ANN parameters
visible in a PR diff. Index building becomes a batch job with content-hash
dedupe and incremental upsert per document; the chunk id is already derived
from a stable document hash and an ordinal.

## 9. Deliberately deferred

Each of these is a known gap with a known shape, not an oversight:

* **Auth, tenancy and per-tenant corpus isolation.** The index has no notion
  of a principal. Document-level ACLs belong in chunk metadata and must be
  applied as a pre-filter in retrieval, never as a post-filter on the answer.
* **Caching.** Query-level and embedding-level caches both pay for themselves;
  neither is implemented, so cost numbers here are worst-case.
* **PDF/HTML/notebook extraction.** One entry in `LOADERS` each.
* **Prompt-injection defence for corpus content.** A corpus document can
  contain instructions. Treating retrieved text as data rather than
  instructions is a generation-layer concern and currently relies only on the
  system prompt.
* **Semantic chunk deduplication across documents.** Near-duplicate removal is
  lexical (Jaccard) and runs only inside a single context assembly.

## 10. Repository map

```
src/archivist/
  types.py        data contracts shared by every layer
  ports.py        the five protocols that make models replaceable
  config.py       every quality-affecting knob, plus a config fingerprint
  text.py         tokenisation and stopwords, shared so metrics and index agree
  app.py          composition root — the only module that picks adapters
  cli.py          ask · research · index · eval · propose-evals
  ingest/         loaders.py, chunking.py
  index/          embeddings.py, vector_store.py, lexical.py, hybrid.py
  retrieve/       query.py, pipeline.py, rerank.py, context.py
  generate/       prompts.py, answerer.py
  agent/          tools.py, loop.py
  evaluation/     dataset.py, metrics.py, harness.py
  obs/            trace.py
  service/        api.py (FastAPI, optional)
corpus/           5 sample documents with checkable facts and a multi-hop chain
evals/            qa.jsonl golden set, committed baselines per strategy
tests/            54 tests, no network, deterministic
```
