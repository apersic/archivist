# Roadmap

A learning path through retrieval-augmented generation, ordered so that each
phase is useful on its own and each one is *checkable*. No calendar estimates:
the exit criterion is a measurement, not a date.

Phases 0–3 are implemented in this repository. Phases 4–7 are the extensions
worth doing next, with the reason each one exists.

---

## Phase 0 — Walking skeleton `done`

Load a corpus, chunk it, retrieve, answer, print citations.

**Why first:** an end-to-end skeleton tells you what the data contracts have
to be. Designing the contracts first and the pipeline second always produces
contracts that do not fit.

**Exit criteria**
- `archivist ask` returns an answer whose citations resolve to real chunks.
- The whole path runs with no network and no API key.

---

## Phase 1 — Evaluation before optimisation `done`

Golden set, retrieval metrics, answer metrics, committed baselines, CI gate.

**Why this early:** without it, every later phase is taste. With it, "hybrid
retrieval helped" becomes a diff. Build the measuring instrument before the
thing you intend to measure, or you will tune prompts to fix an indexing bug.

**Exit criteria**
- Retrieval and generation are scored separately.
- The golden set contains unanswerable cases, and abstention is scored.
- Labels survive a re-chunking (they attach to documents, not chunk ids).
- A quality drop makes `archivist eval` exit non-zero.

**What it caught here:** the heuristic reranker was making answers worse, and
a default change tripped the gate before it could land silently.

---

## Phase 2 — Retrieval that actually works `done`

Structure-aware chunking with heading prefixes, hybrid dense + BM25, RRF
fusion, neighbour expansion, query rewriting and expansion, budgeted context
assembly with near-duplicate removal.

**Exit criteria**
- `source_recall@5` ≥ 0.95 on the golden set.
- An exact-identifier query (`PEL-430`) returns only passages containing it.
- A follow-up question with a pronoun retrieves the right document.

---

## Phase 3 — Grounded generation and bounded agency `done`

Citation parsing and validation, abstention on empty or uncited evidence, a
tool-using research loop with step and search budgets enforced by the loop.

**Exit criteria**
- Invented citation markers are stripped from answers.
- Empty retrieval abstains without calling the model.
- A model that ignores a withdrawn tool cannot exceed the budget.

---

## Phase 4 — Real models, and the cost of them `next`

Swap in hosted embeddings and a chat model; measure the delta on the same
golden set. Add a cross-encoder reranker and settle the open question from
Phase 1. Add caching at the query and embedding level. Record cost and latency
per case in the eval report.

**Why it matters:** this is where the project stops being a toy, and where the
work of Phase 1 pays off — every claim is now a before/after table.

**Exit criteria**
- `answer_match` and `groundedness` reported for offline vs hosted, same cases.
- A cross-encoder ablation that either promotes reranking back to default or
  closes the question with a number.
- p50/p95 latency and cost per answer in the report.
- Cache hit rate measured, and cost per answer re-measured with it on.

---

## Phase 5 — Scale the index `later`

Replace `InMemoryVectorIndex` with pgvector or Qdrant; move indexing to a
batch job with content-hash dedupe and incremental per-document upsert; add a
real lexical engine.

**Exit criteria**
- `source_recall@k` before and after the migration, on the same golden set,
  showing what the ANN parameters cost.
- Re-indexing one changed document does not re-embed the corpus.
- The index survives a process restart.

---

## Phase 6 — Make it safe to expose `later`

Per-tenant corpus isolation as a retrieval **pre-filter** (never a post-filter
on the answer); document-level ACLs in chunk metadata; prompt-injection
handling for corpus content; rate limits; audit log of question → passages →
answer.

**Exit criteria**
- An eval case proving a tenant cannot retrieve another tenant's document.
- An eval case with an injected instruction inside a corpus document that the
  system does not follow.

---

## Phase 7 — Harder questions `later`

Multi-hop chains longer than two steps; aggregation questions ("how many
services write to the warehouse?") which pure retrieval cannot answer and
which need a structured query tool; table-aware extraction; conversational
memory beyond a rewritten question.

**Exit criteria**
- A golden subset tagged `aggregation` that plain RAG fails and the agent
  passes, with both numbers published.

---

## How to use this as a portfolio piece

The repository is designed so that the interesting conversation is available
without a walkthrough of every file:

* **Start with the measurement, not the architecture.** `make eval` prints the
  current state in two seconds, including what is still failing and why.
* **Show a decision that was made by data.** The reranker ablation
  (`--reranker lexical-overlap`) is one flag and it reverses a default.
* **Show a bug the tests caught.** The agent's search budget was enforced only
  by asking the model nicely; `test_search_budget_is_enforced_even_if_the_model_ignores_it`
  is the reason it is enforced by the loop now.
* **Show the seams.** `ports.py` is 60 lines and explains why switching from
  a hash embedder to hosted embeddings is a config change rather than a
  rewrite.
* **Be honest about the floor.** The bundled generator is deterministic and
  weak on purpose. Saying "retrieval is saturated on this corpus and the
  answer layer is the bottleneck, here is the number" is a stronger signal
  than a demo that only works on the happy path.
