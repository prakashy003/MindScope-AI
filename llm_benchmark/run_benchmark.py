"""
Runs the MindScope LLM benchmark.
Iterates: test_sample.csv x 3 prompt strategies x 2 models = 6 JSON result files.

Usage:
    python llm_benchmark/run_benchmark.py
    python llm_benchmark/run_benchmark.py --model claude --strategy zeroshot
"""

import os
import json
import time
import argparse
import pandas as pd
import anthropic
import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)

from llm_benchmark.prompts import STRATEGIES, LABELS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_PATH  = "llm_benchmark/results/test_sample.csv"
RESULTS_DIR  = "llm_benchmark/results"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL    = "gpt-4o-mini"

MAX_RETRIES  = 5
RETRY_DELAY  = 5   # seconds, doubles on each retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_label(raw: str) -> str:
    """Strip whitespace/punctuation and title-case for consistent matching."""
    cleaned = raw.strip().strip(".,\"'").strip()
    for label in LABELS:
        if label.lower() == cleaned.lower():
            return label
    # Partial match fallback
    for label in LABELS:
        if label.lower() in cleaned.lower():
            return label
    return cleaned  # return as-is so we can inspect failures


def parse_cot_response(raw: str) -> tuple[str, str]:
    """Extract category and reasoning from a CoT JSON response."""
    try:
        data = json.loads(raw)
        return normalize_label(data.get("category", "")), data.get("reasoning", "")
    except json.JSONDecodeError:
        # LLM sometimes wraps JSON in markdown — try to extract it
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return normalize_label(data.get("category", "")), data.get("reasoning", "")
            except json.JSONDecodeError:
                pass
        return normalize_label(raw), ""  # fall back to treating full response as label


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------

def call_claude(prompt: str, is_cot: bool) -> tuple[str, str, float]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256 if is_cot else 16,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.time() - t0
            raw = response.content[0].text.strip()
            if is_cot:
                label, reasoning = parse_cot_response(raw)
            else:
                label, reasoning = normalize_label(raw), ""
            return label, reasoning, latency
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  [Claude] Rate limit — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [Claude] Error: {e}")
            time.sleep(RETRY_DELAY)
    return "ERROR", "", 0.0


def call_gpt(prompt: str, is_cot: bool) -> tuple[str, str, float]:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=256 if is_cot else 16,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.time() - t0
            raw = response.choices[0].message.content.strip()
            if is_cot:
                label, reasoning = parse_cot_response(raw)
            else:
                label, reasoning = normalize_label(raw), ""
            return label, reasoning, latency
        except openai.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  [GPT] Rate limit — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [GPT] Error: {e}")
            time.sleep(RETRY_DELAY)
    return "ERROR", "", 0.0


MODEL_CALLERS = {
    "claude": call_claude,
    "gpt":    call_gpt,
}

# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run(model_name: str, strategy_name: str, df: pd.DataFrame):
    out_path = f"{RESULTS_DIR}/{model_name}_{strategy_name}.json"

    # Resume: load already-completed rows
    completed = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for item in json.load(f):
                completed[item["index"]] = item

    caller   = MODEL_CALLERS[model_name]
    template = STRATEGIES[strategy_name]
    is_cot   = strategy_name == "cot"
    results  = list(completed.values())

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  Strategy: {strategy_name}")
    print(f"Completed so far: {len(completed)}/{len(df)}")
    print(f"{'='*60}")

    for idx, row in df.iterrows():
        if idx in completed:
            continue

        prompt = template.format(text=row["statement"])
        predicted, reasoning, latency = caller(prompt, is_cot)

        entry = {
            "index":      idx,
            "statement":  row["statement"],
            "true_label": row["status"],
            "predicted":  predicted,
            "reasoning":  reasoning,
            "latency_s":  round(latency, 3),
        }
        results.append(entry)
        completed[idx] = entry

        correct = "✓" if predicted == row["status"] else "✗"
        print(f"  [{idx:>3}] {correct} true={row['status']:<22} pred={predicted:<22} ({latency:.2f}s)")

        # Save after every row
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["claude", "gpt", "all"], default="all")
    parser.add_argument("--strategy", choices=["zeroshot", "fewshot", "cot", "all"], default="all")
    args = parser.parse_args()

    df = pd.read_csv(SAMPLE_PATH)

    models     = ["claude", "gpt"]     if args.model    == "all" else [args.model]
    strategies = ["zeroshot", "fewshot", "cot"] if args.strategy == "all" else [args.strategy]

    for model in models:
        for strategy in strategies:
            run(model, strategy, df)

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
