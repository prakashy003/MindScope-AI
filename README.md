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

The successful **MentalBERT** model was deployed via a complete **Flask web application**. Using HTML, CSS, and JS, a beautiful frontend interface was constructed for the model. Users may now type sentences reflecting different emotional or psychological states into the frontend and get an accurate mental health tag returned in real-time.
