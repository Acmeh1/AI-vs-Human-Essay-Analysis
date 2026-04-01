# 🤖 Distinguishing AI-Generated from Human-Written Text

> A full data analysis & machine learning pipeline that detects AI-written essays with **95.7% accuracy** using only stylometric text features — no embeddings, no transformers.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0.987_AUC-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project analyzes **487,235 student essays** (62.8% human, 37.2% AI-generated) and builds a classifier to distinguish between them using hand-crafted linguistic features.

The key insight: **writing style leaves a fingerprint**. AI text tends to use longer, harder words and score higher on readability grade levels — while human text is longer, more conversational, and uses simpler vocabulary.

---

## 📊 Dataset

- **Source:** [Kaggle — Distinguishing AI-Generated from Human-Written Text](https://www.kaggle.com/wissam2000/distinguishing-ai-generated-from-human-written-tex)
- **Size:** 487,235 essays
- **Classes:** Human (305,797 · 62.8%) / AI (181,438 · 37.2%)

---

## 🗂️ Repository Structure

```
├── notebook/
│   └── distinguishing-ai-generated-from-human-written-tex.ipynb
├── images/
│   ├── step1_distribution.png
│   ├── step2_boxplots.png
│   ├── step2_comparison.png
│   ├── step2_distributions.png
│   ├── step3_correlation.png
│   ├── step3_readability.png
│   ├── step4_top_words.png
│   ├── step5_significant_features.png
│   ├── step6_confusion.png
│   ├── step6_feature_importance.png
│   └── step6_roc.png
└── README.md
```

---

## ⚙️ Pipeline — 6 Steps

| Step | Description |
|------|-------------|
| **1 — Data Loading** | Load & explore dataset, check class balance |
| **2 — Feature Engineering** | Extract 25+ stylometric features per essay |
| **3 — EDA & Readability** | Compare distributions, correlation with label |
| **4 — Vocabulary Analysis** | Top content words per class |
| **5 — Statistical Tests** | Cohen's d effect sizes for key features |
| **6 — ML Classification** | Train & evaluate 3 classifiers |

---

## 🧬 Features Engineered (25+)

**Length & structure**
`char_count` · `word_count` · `unique_words` · `sentence_count` · `paragraph_count` · `avg_word_length` · `avg_sentence_length` · `avg_paragraph_length`

**Vocabulary richness**
`lexical_diversity` · `hapax_ratio` · `long_word_ratio`

**Punctuation**
`comma_count` · `period_count` · `exclamation_count` · `question_count` · `semicolon_count` · `punctuation_density` · `comma_per_sentence`

**Readability scores**
`flesch_reading_ease` · `flesch_kincaid_grade` · `gunning_fog` · `smog_index` · `automated_readability` · `coleman_liau` · `dale_chall`

---

## 📈 Results

### Key Metric Differences (Human vs AI)

| Metric | Human | AI |
|--------|-------|----|
| Avg essay length | 419 words | 344 words |
| Avg sentence count | 21.6 | 18.7 |
| Avg word length | 4.45 chars | 4.98 chars |
| Vocabulary variety (TTR) | 0.44 | 0.47 |
| Flesch Reading Ease | 64 | 47 |
| Flesch-Kincaid Grade | 9.5 | 11.5 |

### Step 1 — Class Distribution

![Distribution](images/step1_distribution.png)

### Step 2 — Feature Engineering

![Comparison](images/step2_comparison.png)
![Box Plots](images/step2_boxplots.png)
![Distributions](images/step2_distributions.png)

### Step 3 — Readability & Correlation

![Readability](images/step3_readability.png)
![Correlation](images/step3_correlation.png)

> **Top positive correlators with AI:** `long_word_ratio` (+0.58), `avg_word_length` (+0.57), `coleman_liau` (+0.54)
> **Top negative correlators:** `flesch_reading_ease` (−0.48), `word_count` (−0.33)

### Step 4 — Vocabulary Analysis

![Top Words](images/step4_top_words.png)

> Human essays favor casual words: *"just", "think", "car"*
> AI essays favor formal words: *"electoral", "important", "states", "usage"*

### Step 5 — Statistical Significance

![Significant Features](images/step5_significant_features.png)

| Feature | Cohen's d | Effect Size |
|---------|-----------|-------------|
| Avg Word Length | −1.41 | Large |
| Readability (Flesch) | −0.95 | Large |
| Lexical Diversity | −0.42 | Small-Medium |
| Essay Length | 0.49 | Small-Medium |
| Sentence Count | 0.35 | Small |
| Avg Sentence Length | 0.26 | Small |

### Step 6 — ML Classification

![ROC Curves](images/step6_roc.png)
![Confusion Matrices](images/step6_confusion.png)
![Feature Importance](images/step6_feature_importance.png)

| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | 88.0% | 0.952 |
| XGBoost | 94.0% | 0.987 |
| **Random Forest** ✅ | **95.7%** | **0.992** |

**Top 3 most important features (Random Forest):**
1. `long_word_ratio`
2. `avg_word_length`
3. `coleman_liau`

---

## 🛠️ Tech Stack

- **Python 3.11**
- `pandas` / `numpy` — data manipulation
- `textstat` — readability scores
- `scikit-learn` — Random Forest, Logistic Regression
- `xgboost` — gradient boosting
- `matplotlib` / `seaborn` — visualizations

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Acmeh1/NCM_data.git
cd NCM_data

# 2. Install dependencies
pip install pandas numpy textstat scikit-learn xgboost matplotlib seaborn

# 3. Download the dataset
kaggle kernels output wissam2000/distinguishing-ai-generated-from-human-written-tex -p ./data

# 4. Open the notebook
jupyter notebook notebook/distinguishing-ai-generated-from-human-written-tex.ipynb
```

---

## 💡 Key Takeaway

> You don't need a giant language model to detect AI-generated text.
> 25 simple stylometric features + Random Forest = **95.7% accuracy, AUC 0.992**.
> Style leaves a fingerprint.

---

## 📄 License

MIT — feel free to use, adapt, and share.
