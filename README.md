# Twitter Sentiment Analysis

Sentiment classification on Twitter data using machine learning — built as part of a university group assignment (CT107-3-3-TXSA, APU).

## Overview

This project covers two parts:

**Part A — NLP Fundamentals**
- Word tokenisation (Regex, NLTK, TextBlob, spaCy)
- Stop word & punctuation removal
- Word stemming (Regex, Porter, Lancaster)
- POS tagging & parse tree generation
- Bigram language model (unsmoothed & Laplace-smoothed)

**Part B — Sentiment Classification**
- Dataset: 74,682 Twitter records (Positive / Negative / Neutral / Irrelevant)
- Models: Multinomial Naïve Bayes, Support Vector Machine, Random Forest
- Hyperparameter tuning via GridSearchCV (5-fold cross-validation)

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Multinomial Naïve Bayes | 85% | 86% | 85% | 85% |
| Support Vector Classifier | 96% | 96% | 95% | 95% |
| **Random Forest** | **97%** | **97%** | **97%** | **97%** |

Random Forest performed best after hyperparameter tuning (n_estimators=200, max_depth=None, min_samples_split=10).

---

## Repository Structure

```
twitter-sentiment-analysis/
├── notebooks/
│   ├── TXSA_assignment_PartA.ipynb
│   └── TXSA_assignment_PartB.ipynb
├── dataset/
│   └── DATASET PART B/
└── README.md
```

---

## Tech Stack

- Python, Jupyter Notebook
- NLTK, TextBlob, spaCy
- scikit-learn (MNB, SVC, RandomForest, GridSearchCV)
- pandas, numpy
