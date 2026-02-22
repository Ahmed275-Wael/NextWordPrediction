"""
View Pre-Computed Test Results
==============================
Displays the results from tester.py without needing model checkpoints or data.
Run this if you don't have the large model/data files.

Usage:
    python view_results.py                     # Pretty-print summary table
    python view_results.py --json              # Print raw JSON
    python view_results.py --file results.json # Use a custom results file
"""

import json
import argparse
import sys
from pathlib import Path


DEFAULT_RESULTS = Path(__file__).parent / "test_results.json"


def load_results(path: Path) -> list:
    if not path.exists():
        print(f"ERROR: Results file not found: {path}")
        print("Run tester.py first to generate results, or ensure test_results.json is present.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(results: list) -> None:
    # Sort by perplexity (best first)
    ranked = sorted(results, key=lambda r: r["perplexity"])

    header = (
        f"{'Rank':<5} {'Model':<42} {'Params':>8} {'PPL':>8} {'Acc%':>7} "
        f"{'Top5%':>7} {'Tok/s':>10}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("SHAKESPEARE MODEL TEST RESULTS")
    print("=" * len(header))
    print(f"  Models evaluated: {len(ranked)}")
    print(f"  Best perplexity:  {ranked[0]['perplexity']:.1f} ({ranked[0]['display_name']})")
    print(f"  Best accuracy:    {max(r['accuracy'] for r in ranked):.2f}%")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    for i, r in enumerate(ranked, 1):
        params_str = f"{r['total_params'] / 1e6:.1f}M"
        marker = " *" if i == 1 else ""
        print(
            f"{i:<5} {r['display_name']:<42} {params_str:>8} "
            f"{r['perplexity']:>8.1f} {r['accuracy']:>6.2f}% "
            f"{r['top5_accuracy']:>6.2f}% {r['tokens_per_sec']:>10,.0f}{marker}"
        )

    print(sep)
    print()

    # Key comparisons
    best = ranked[0]
    worst = ranked[-1]
    improvement = (1 - best["perplexity"] / worst["perplexity"]) * 100

    print("Key Comparisons:")
    print(f"  Best vs Worst:  {worst['perplexity']:.1f} -> {best['perplexity']:.1f} PPL "
          f"({improvement:.1f}% reduction)")

    # Find specific models for comparisons
    names = {r["checkpoint"]: r for r in ranked}

    if "best_model.pt" in names and "best_model_bpe.pt" in names:
        wl = names["best_model.pt"]
        bpe = names["best_model_bpe.pt"]
        print(f"  Word vs BPE:    {wl['perplexity']:.1f} -> {bpe['perplexity']:.1f} PPL "
              f"({(1 - bpe['perplexity'] / wl['perplexity']) * 100:.1f}% reduction)")

    if "best_model_bpe.pt" in names and "finetuned_shakespeare_v4.pt" in names:
        scratch = names["best_model_bpe.pt"]
        ft = names["finetuned_shakespeare_v4.pt"]
        print(f"  Scratch vs TL:  {scratch['perplexity']:.1f} -> {ft['perplexity']:.1f} PPL "
              f"({(1 - ft['perplexity'] / scratch['perplexity']) * 100:.1f}% reduction)")

    if "best_model_lstm.pt" in names and "best_model_bpe.pt" in names:
        lstm = names["best_model_lstm.pt"]
        tf = names["best_model_bpe.pt"]
        print(f"  LSTM vs Transformer (scratch): {lstm['perplexity']:.1f} ({lstm['total_params']/1e6:.1f}M) "
              f"vs {tf['perplexity']:.1f} ({tf['total_params']/1e6:.1f}M)")

    print()


def print_json(results: list) -> None:
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="View pre-computed Shakespeare model test results",
    )
    parser.add_argument(
        "--file", type=str, default=str(DEFAULT_RESULTS),
        help=f"Path to results JSON file (default: {DEFAULT_RESULTS.name})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print raw JSON instead of formatted table",
    )
    args = parser.parse_args()

    results = load_results(Path(args.file))

    if args.json:
        print_json(results)
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
