"""Command line entry point.

    archivist ask "question"        plain RAG, one retrieval
    archivist research "question"   agentic loop, shows its steps
    archivist index                 report what the corpus chunks into
    archivist eval                  run the golden set, fail on regression
    archivist propose-evals         draft golden cases from the corpus

Argparse rather than a CLI framework: no dependency, and the commands map
one-to-one onto the layers so the CLI doubles as documentation of the design.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archivist.app import build
from archivist.config import Settings
from archivist.evaluation.dataset import load_cases, propose_cases, save_cases
from archivist.evaluation.harness import compare, evaluate, save_report
from archivist.types import Answer

RERANKERS = ("identity", "lexical-overlap", "cross-encoder")


def _verbose(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "verbose", False))


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    corpus = getattr(args, "corpus", None)
    if corpus:
        overrides["corpus_dir"] = Path(corpus)
    retrieval: dict[str, object] = {}
    if getattr(args, "top_k", None):
        retrieval["rerank_limit"] = args.top_k
    if getattr(args, "reranker", None):
        overrides["providers"] = {"reranker": args.reranker}
    if retrieval:
        overrides["retrieval"] = retrieval
    return Settings.from_env(overrides)


def _print_answer(answer: Answer, verbose: bool) -> None:
    print(answer.text)
    if answer.citations:
        print("\nsources:")
        for citation in answer.citations:
            print(f"  [{citation.marker}] {citation.location}  ({citation.chunk_id})")
    if verbose and answer.contexts:
        print("\nretrieved:")
        for scored in answer.contexts:
            snippet = " ".join(scored.chunk.text.split())[:140]
            print(f"  {scored.score:6.3f} {scored.retriever:<24} {snippet}")


def cmd_ask(args: argparse.Namespace) -> int:
    app = build(_settings(args))
    _print_answer(app.ask(args.question), _verbose(args))
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    app = build(_settings(args))
    verbose = _verbose(args)
    run = app.agent.run(args.question)
    for step in run.steps:
        print(f"step {step.index}: {step.tool}({step.arguments})")
        if verbose:
            print("  " + step.observation.replace("\n", "\n  "))
    print()
    _print_answer(run.answer, verbose)
    if verbose:
        print("\ntrace:")
        print(run.trace.render())
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    app = build(_settings(args))
    chunks = app.vector_index.all_chunks()
    sources = sorted({chunk.source for chunk in chunks})
    tokens = [chunk.token_estimate for chunk in chunks]
    print(f"documents: {len(sources)}")
    print(f"chunks:    {len(chunks)}")
    print(f"tokens:    min={min(tokens)} mean={sum(tokens) // len(tokens)} max={max(tokens)}")
    print(f"embedder:  {app.embedder.name} ({app.embedder.dimensions} dims)")
    for source in sources:
        count = sum(1 for chunk in chunks if chunk.source == source)
        print(f"  {count:3d}  {source}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    app = build(_settings(args))
    cases = load_cases(Path(args.cases))
    strategies = {
        "plain-rag": lambda question, history: app.ask(question, history=history),
        "agentic": lambda question, history: app.research(question, history=history),
    }
    exit_code = 0
    for name in args.strategy:
        report = evaluate(app, cases, strategy=strategies[name], strategy_name=name, k=args.k)
        print(report.render())
        print()
        if args.report:
            save_report(report, Path(args.report).with_suffix(f".{name}.json"))
        baseline = Path(args.baselines) / f"{name}.json"
        if args.update_baselines:
            save_report(report, baseline)
            print(f"baseline updated: {baseline}")
            continue
        regressions = compare(report, baseline, tolerance=args.tolerance)
        if regressions:
            print(f"REGRESSION vs {baseline}:")
            for regression in regressions:
                print(f"  {regression}")
            exit_code = 1
    return exit_code


def cmd_propose_evals(args: argparse.Namespace) -> int:
    app = build(_settings(args))
    drafts = propose_cases(app.vector_index.all_chunks(), per_source=args.per_source)
    save_cases(drafts, Path(args.out))
    print(f"wrote {len(drafts)} draft cases to {args.out} - review them by hand before use")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared flags live on a parent parser so they are accepted on either side
    # of the subcommand: `archivist -v ask ...` and `archivist ask ... -v`.
    # They default to SUPPRESS because a subparser's defaults are applied last
    # and would otherwise silently overwrite a value given before the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--corpus", default=argparse.SUPPRESS, help="corpus directory (default: ./corpus)"
    )
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(prog="archivist", description=__doc__, parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="plain RAG answer", parents=[common])
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, help="passages kept after reranking")
    ask.add_argument("--reranker", choices=RERANKERS)
    ask.set_defaults(func=cmd_ask)

    research = subparsers.add_parser(
        "research", help="agentic answer with visible steps", parents=[common]
    )
    research.add_argument("question")
    research.add_argument("--top-k", type=int)
    research.add_argument("--reranker", choices=RERANKERS)
    research.set_defaults(func=cmd_research)

    index = subparsers.add_parser("index", help="summarise the chunked corpus", parents=[common])
    index.set_defaults(func=cmd_index)

    evaluate_cmd = subparsers.add_parser("eval", help="run the golden set", parents=[common])
    evaluate_cmd.add_argument("--cases", default="evals/qa.jsonl")
    evaluate_cmd.add_argument("--baselines", default="evals/baselines")
    evaluate_cmd.add_argument("--report", help="write a JSON report to this path prefix")
    evaluate_cmd.add_argument(
        "--strategy", nargs="+", choices=["plain-rag", "agentic"], default=["plain-rag"]
    )
    evaluate_cmd.add_argument("--k", type=int, default=5)
    # Ablation flags: the same levers as `ask`, so a claim like "reranking is
    # worth it" is a measured comparison rather than an opinion.
    evaluate_cmd.add_argument("--top-k", type=int)
    evaluate_cmd.add_argument("--reranker", choices=RERANKERS)
    evaluate_cmd.add_argument("--tolerance", type=float, default=0.02)
    evaluate_cmd.add_argument("--update-baselines", action="store_true")
    evaluate_cmd.set_defaults(func=cmd_eval)

    propose = subparsers.add_parser(
        "propose-evals", help="draft golden cases from the corpus", parents=[common]
    )
    propose.add_argument("--out", default="evals/drafts.jsonl")
    propose.add_argument("--per-source", type=int, default=1)
    propose.set_defaults(func=cmd_propose_evals)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
