# Archivist

An agentic RAG system over a private document corpus, built to be measured.

Ask a question, get an answer with citations you can open — or an explicit
refusal when the corpus does not contain the answer. Retrieval quality and
answer quality are measured separately by a committed eval harness, so every
tuning claim in this repository is a number someone can re-run.

It runs with **zero dependencies and no API key**: the default embedder,
indexes and generator are deterministic local implementations behind the same
ports as the hosted ones. Swap in OpenAI embeddings and `gpt-4o-mini` with two
environment variables and re-run the same evals to see what changed.

```bash
git clone <this repo> && cd archivist
python3 -m archivist.cli index     # 5 documents, 27 chunks
python3 -m archivist.cli ask "What error code does Pelican return when an API key exceeds its rate limit?"
```

```
Exceeding the limit returns HTTP 429 with the error code `PEL-429` and a
`Retry-After` header. [S1]

sources:
  [S1] Pelican Ingestion Service / Rate limits  (doc-e8d975ade0:0003)
  [S2] On-Call Runbook / Handling PEL-429 rate limiting  (doc-7afa88e5a1:0001)
```

## Why this project exists

It is a vehicle for learning retrieval-augmented generation properly, and for
showing that the learning happened. The interesting parts of RAG are not the
`pip install`; they are chunking strategy, hybrid retrieval, grounded
generation, bounded agency, and above all **evaluation**. So the harness is
not an afterthought here — it is the component that decides what ships, and
it has already overruled one of the defaults (see below).

[`ARCHITECTURE.md`](ARCHITECTURE.md) explains the design and the reasoning.
[`ROADMAP.md`](ROADMAP.md) is the learning path, phase by phase, with exit
criteria you can check yourself.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core has no runtime dependencies
pip install -e ".[dev,openai,service]"   # hosted models + HTTP API
```

Python 3.11+.

## Commands

| Command | What it does |
|---|---|
| `archivist index` | Report what the corpus chunks into: documents, chunks, token distribution |
| `archivist ask "…"` | Plain RAG: one retrieval, one generation. Add `-v` for the retrieved passages |
| `archivist research "…"` | Agentic: bounded tool loop, prints each step and the trace with `-v` |
| `archivist eval` | Run the golden set, print metrics, exit non-zero on regression |
| `archivist eval --strategy plain-rag agentic` | Compare both strategies side by side |
| `archivist eval --reranker lexical-overlap` | Ablate one stage |
| `archivist propose-evals` | Draft golden cases from the corpus for a human to correct |

Use your own corpus with `--corpus /path/to/docs` (Markdown, text, reStructured
Text today; one `LOADERS` entry per new format).

## Swapping in real models

```bash
export OPENAI_API_KEY=sk-…
export ARCHIVIST_EMBEDDER=openai        # text-embedding-3-small
export ARCHIVIST_LLM=openai             # gpt-4o-mini
export ARCHIVIST_MODEL=gpt-4o-mini
archivist eval --strategy plain-rag agentic
```

Nothing in the pipeline changes: `app.py` is the only module that picks
adapters, and everything else talks to the five protocols in `ports.py`.

## Measured state

Bundled corpus (5 documents, 27 chunks), 12-case golden set, offline
extractive generator — the floor, not the ceiling:

| strategy | source_recall@5 | ndcg@5 | groundedness | citation_support | answer_match | abstention_acc |
|---|---|---|---|---|---|---|
| plain-rag | 1.000 | 0.993 | 1.000 | 1.000 | 0.833 | 0.917 |
| agentic | 1.000 | 0.944 | 1.000 | 1.000 | 0.833 | 0.917 |

Retrieval is saturated on a corpus this size; the answer layer is the
bottleneck. Two findings worth more than the table:

* **The reranker was demoted by its own ablation.** `lexical-overlap`
  reranking lowered `answer_match` from 0.833 to 0.750, so the default is now
  `identity`. The stage is still there behind `--reranker`; a cross-encoder is
  expected to win, and until that is measured it is not the default.
* **The regression gate works.** Changing that default made
  `archivist eval` exit non-zero against the committed baselines until the
  baselines were updated deliberately, in the same commit.

Baselines live in `evals/baselines/*.json` and are committed, so a tuning PR
shows the config change and its measured effect in the same diff.

## Development

```bash
make test     # 54 tests, no network, ~0.5s
make lint     # ruff
make types    # mypy --strict
make eval     # golden set + regression gate
make check    # all of the above
```

Without `make` (Windows PowerShell, for example), the targets are one command each:

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
python -m archivist.cli eval --strategy plain-rag agentic
```

The tests are deterministic because the default LLM is. They assert on system
properties that hold for any model: citations resolve to real chunks, invented
citation markers are stripped, empty retrieval abstains without calling the
model, agent budgets are enforced even when the model ignores them.

## Repository map

```
src/archivist/   ingest · index · retrieve · generate · agent · evaluation · obs · service
corpus/          sample documents with checkable facts and a multi-hop chain
evals/           golden set + committed baselines
tests/           54 deterministic tests
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) §10 for the file-by-file map.
