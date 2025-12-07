# Arabic-Generated-Abstracts-Project


## Project Overview

This repository contains the code and resources for detecting **AI-generated vs. human-written Arabic text** using machine learning and deep learning models built on top of transformer embeddings.


---

##  Task Description

The core objective is formulated as a **binary classification problem**:


Each Arabic abstract is first preprocessed, then encoded into a high-dimensional embedding vector, and finally classified using one of several trained models. This project addresses the rising challenge of identifying automatically generated Arabic academic text in a variety of contexts.

---

## 📚 Dataset

We utilize a large, custom-curated dataset of **41,940 Arabic research abstracts**, composed of:
* Human-written abstracts collected from academic sources.
* AI-generated abstracts created using multiple Large Language Models (LLMs)
### Dataset Summary

| Split | Count |
| :--- | :---: |
| Training | 29,358 |
| Validation/Test | 6,291 |
| **Total Samples** | **41,940** |
| **Classes** | **0 (AI), 1 (Human)** |

The dataset is **balanced**, ensuring stable and fair model performance across both classes.





##  Machine Learning Models Results

Performance metrics on the held-out test set using Sentence-Transformer embeddings:

| Model | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **SVM** | **0.94** | **0.92** | **0.91** | **0.91** |
| XGBoost | 0.97 | 0.96 | 0.96 | 0.96 |
| Logistic Regression | 0.96 | 0.95 | 0.93 | 0.94 |

---

##  Deep Learning Results

Performance metrics on the held-out test set:

| Model | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| FFNN (BERT Embeddings) | 0.870 | 0.86 | 0.87 | 0.86 |
