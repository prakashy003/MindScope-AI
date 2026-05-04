"""
Computes metrics for all 6 LLM benchmark runs and prints a comparison table
against MentalBERT's known results.

Usage:
    python llm_benchmark/evaluate.py
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from llm_benchmark.prompts import STRATEGIES

RESULTS_DIR = "llm_benchmark/results"
LABELS = ["Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress", "Personality disorder"]

# Published token pricing (per 1M tokens)
PRICING = {
    "claude": {"input": 0.80, "output": 4.00},
    "gpt":    {"input": 0.15, "output": 0.60},
}

# Approximate output tokens per strategy
OUTPUT_TOKENS = {"zeroshot": 5, "fewshot": 5, "cot": 200}


def estimate_cost_per_1k(model: str, strategy: str, results: list) -> float:
    """Estimate cost per 1000 predictions from prompt length + output token estimate."""
    sample_text = results[0]["statement"] if results else ""
    prompt = STRATEGIES[strategy].format(text=sample_text)
    input_tokens  = len(prompt) // 4
    output_tokens = OUTPUT_TOKENS[strategy]
    price = PRICING[model]
    cost_per_pred = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return round(cost_per_pred * 1000, 3)


def compute_metrics(results: list) -> dict:
    true   = [r["true_label"] for r in results]
    pred   = [r["predicted"]  for r in results]
    latencies = [r["latency_s"] for r in results]

    macro_f1  = f1_score(true, pred, average="macro",    labels=LABELS, zero_division=0)
    accuracy  = accuracy_score(true, pred)
    per_class = f1_score(true, pred, average=None,       labels=LABELS, zero_division=0)
    pd_f1     = per_class[LABELS.index("Personality disorder")]
    avg_lat   = sum(latencies) / len(latencies)

    return {
        "accuracy":  round(accuracy * 100, 1),
        "macro_f1":  round(macro_f1, 4),
        "pd_f1":     round(pd_f1, 4),
        "latency_s": round(avg_lat, 2),
        "per_class": {LABELS[i]: round(v, 4) for i, v in enumerate(per_class)},
    }


def load_run(model: str, strategy: str) -> list:
    path = os.path.join(RESULTS_DIR, f"{model}_{strategy}.json")
    with open(path) as f:
        return json.load(f)


def main():
    rows = []

    # MentalBERT — known results from notebook 07
    rows.append({
        "Model":    "MentalBERT (fine-tuned)",
        "Strategy": "—",
        "Accuracy": "84.2%",
        "Macro F1": "0.8380",
        "F1 (Personality disorder)": "0.8300",
        "Avg Latency": "~local",
        "Cost / 1k":   "~$0.00",
    })

    runs = [
        ("claude", "zeroshot"),
        ("claude", "fewshot"),
        ("claude", "cot"),
        ("gpt",    "zeroshot"),
        ("gpt",    "fewshot"),
        ("gpt",    "cot"),
    ]

    for model, strategy in runs:
        results = load_run(model, strategy)
        m       = compute_metrics(results)
        cost    = estimate_cost_per_1k(model, strategy, results)
        model_label = "Claude Haiku" if model == "claude" else "GPT-4o-mini"

        rows.append({
            "Model":    model_label,
            "Strategy": strategy,
            "Accuracy": f"{m['accuracy']}%",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "F1 (Personality disorder)": f"{m['pd_f1']:.4f}",
            "Avg Latency": f"{m['latency_s']}s",
            "Cost / 1k":   f"${cost:.3f}",
        })

    df = pd.DataFrame(rows)

    # --- Terminal table ---
    print("\n" + "=" * 90)
    print("  MINDSCOPE LLM BENCHMARK — RESULTS")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # --- Per-class F1 detail for LLM runs ---
    print("\n\nPER-CLASS F1 DETAIL\n" + "-" * 70)
    detail_rows = []
    for model, strategy in runs:
        results = load_run(model, strategy)
        m = compute_metrics(results)
        model_label = "Claude Haiku" if model == "claude" else "GPT-4o-mini"
        row = {"Model": model_label, "Strategy": strategy}
        row.update(m["per_class"])
        detail_rows.append(row)

    # Prepend MentalBERT
    mentalbert_row = {
        "Model": "MentalBERT", "Strategy": "—",
        "Normal": 0.9559, "Depression": 0.7848, "Suicidal": 0.7515,
        "Anxiety": 0.8715, "Bipolar": 0.8870, "Stress": 0.7850,
        "Personality disorder": 0.8300,
    }
    detail_rows.insert(0, mentalbert_row)
    detail_df = pd.DataFrame(detail_rows)
    print(detail_df.to_string(index=False))

    # --- Markdown table for README ---
    md_path = os.path.join(RESULTS_DIR, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("## LLM Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n### Per-Class F1\n\n")
        f.write(detail_df.to_markdown(index=False))
        f.write("\n")
    print(f"\nMarkdown table saved to {md_path}")


if __name__ == "__main__":
    main()
