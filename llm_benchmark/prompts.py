"""
Three prompt strategies for the MindScope LLM benchmark.
All templates use {text} as the placeholder for the input statement.
"""

LABELS = [
    "Anxiety", "Depression", "Suicidal",
    "Stress", "Bipolar", "Personality disorder", "Normal"
]

LABEL_LIST_STR = ", ".join(LABELS)

# ---------------------------------------------------------------------------
# Strategy 1 — Zero-shot
# ---------------------------------------------------------------------------

ZERO_SHOT = (
    "You are a mental health text classifier. "
    "Classify the following text into exactly one of these categories: "
    f"[{LABEL_LIST_STR}]. "
    "Respond with only the category name, nothing else.\n\n"
    "Text: {text}"
)

# ---------------------------------------------------------------------------
# Strategy 2 — Few-shot (1 labeled example per class, from training set)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = [
    ("Normal",
     "i especially love the night air when it rains"),
    ("Depression",
     "i feel like a loser 18 male still virgin akward af geez"),
    ("Suicidal",
     "i am just done its almost 3am and i am about to cut my wrist open "
     "idk why i should stay any longer i am ready to end it all"),
    ("Anxiety",
     "overthinking how can i control my thoughts instead of letting them overwhelm me"),
    ("Bipolar",
     "what kind of image would you use to illustrate being bipolar "
     "and what description would you use with that image"),
    ("Stress",
     "nausea my dissertation is due in 2 weeks and lately i ve had nausea "
     "all day it s surely stress related any management tips thanks"),
    ("Personality disorder",
     "this is avpd contained in a song stumbled over this and thought it would resonate lol"),
]

_examples_block = "\n".join(
    f"Text: {text}\nCategory: {label}"
    for label, text in _FEW_SHOT_EXAMPLES
)

FEW_SHOT = (
    "You are a mental health text classifier. "
    "Classify the following text into exactly one of these categories: "
    f"[{LABEL_LIST_STR}]. "
    "Respond with only the category name, nothing else.\n\n"
    "Here are some examples:\n\n"
    f"{_examples_block}\n\n"
    "Now classify this:\n"
    "Text: {text}\n"
    "Category:"
)

# ---------------------------------------------------------------------------
# Strategy 3 — Chain-of-thought + structured JSON output
# ---------------------------------------------------------------------------

COT = (
    "You are a mental health text classifier. "
    f"The valid categories are: [{LABEL_LIST_STR}].\n\n"
    "Analyze the following text step by step:\n"
    "1. Identify emotional cues present in the text.\n"
    "2. Identify linguistic patterns associated with specific mental health conditions.\n"
    "3. Determine the single most likely category.\n\n"
    "Respond in valid JSON only, with this exact structure:\n"
    '{{"reasoning": "<your step-by-step analysis>", "category": "<category name>"}}\n\n'
    "Text: {text}"
)

# ---------------------------------------------------------------------------
# Convenience registry — used by run_benchmark.py
# ---------------------------------------------------------------------------

STRATEGIES = {
    "zeroshot": ZERO_SHOT,
    "fewshot":  FEW_SHOT,
    "cot":      COT,
}
