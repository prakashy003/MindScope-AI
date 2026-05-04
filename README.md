# MindScope AI

## Overview
**MindScope AI** is an NLP-based system designed to detect and classify mental health conditions from text inputs.  
Using the *Sentiment Analysis for Mental Health* dataset from Hugging Face, it explores how linguistic cues reflect emotional and psychological states.

The project compared multiple modeling paradigms—**TF-IDF + SVM**, **Word2Vec + LSTM**, and **Fine-tuned Transformers (MentalBERT)**—to determine which approach most effectively captures nuanced mental health expressions.

## Model Comparison

- **TF-IDF + SVM**: Achieved a macro F1 score of **0.71**. This traditional approach offered solid baseline performance but struggled with minor classes and complex sentence structures due to a lack of contextual understanding.
- **Word2Vec + LSTM**: Attained a macro F1 score of **0.69**. While better at understanding sequence patterns, it encountered difficulties with nuanced emotions compared to transformer models.
- **MentalBERT (Winner)**: Outperformed both baseline models significantly, scoring a test accuracy of **84.2%** and a macro F1 score of **0.84**. This domain-adapted model perfectly matched infrequent targets like "Personality Disorder" and showed profound context awareness.

## Deployment

The successful **MentalBERT** model was deployed via a complete **Flask web application**. Users may now type sentences reflecting different emotional or psychological states into the frontend and get an accurate mental health tag returned in real-time.
The interface allows users to quickly test the model with natural language inputs while the backend processes the text using the trained MentalBERT classifier.

Web Application Interface:

<p align="center">
<img width="673" height="904" alt="image" src="https://github.com/user-attachments/assets/9790929e-65c7-48e6-8847-f0ff5d03e136" />
</p>
Example of the MindScope AI web interface where users input text and receive mental health classification results.

In the example above, the user enters the sentence:

> "I just feel completely empty. I don't see the point in waking up anymore."

The **MentalBERT model** analyzes the text and classifies it as **Suicidal** with a **confidence score of 99.7%**.  
The interface displays the predicted label, the model’s confidence score, and a probability breakdown across all supported classes (*Suicidal, Personality Disorder, Stress, Bipolar, Normal, Depression, Anxiety*).

This visualization helps users quickly understand how the model interprets emotional cues in the input text.

> **Disclaimer:** This tool is for research and demonstration purposes only and does not replace professional medical advice.

---

## Future Plans — LLM Benchmark Extension

The next planned extension is a **zero-shot LLM benchmark** that directly compares the fine-tuned MentalBERT model against general-purpose large language models (Claude and GPT) on the same labeled test set.

The benchmark will evaluate three prompt engineering strategies — zero-shot, few-shot, and chain-of-thought — across two cost-efficient models (`claude-haiku-4-5` and `gpt-4o-mini`). Each run will be measured on accuracy, macro F1, per-class F1 (with special attention to minority classes like Personality Disorder), latency, and cost per 1,000 predictions.

The goal is to answer a concrete question: *does a domain-adapted fine-tuned model outperform a general-purpose LLM on mental health text classification — especially on rare, clinically significant labels?* The findings will be documented in a comparison table and a short write-up covering trade-offs between model type, prompt strategy, accuracy, and inference cost.
