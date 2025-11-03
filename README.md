# SEO Content Quality & Duplicate Detector

A machine learning pipeline for automated SEO content analysis that detects near-duplicate pages and scores content quality using NLP techniques. The system analyzes HTML content to classify pages into High/Medium/Low quality tiers and identifies duplicate content using semantic embeddings.

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Kiranrossi/seo-content-detector
cd seo-content-detector

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebooks/seo_pipeline.ipynb
```

---

## Quick Start

1. Place `data.csv` (from Kaggle) in the `data/` folder
2. Run all cells in `seo_pipeline.ipynb` (Runtime: ~5-10 minutes)
3. Review outputs in `data/` folder: `extracted_content.csv`, `features.csv`, `duplicates.csv`

The notebook automatically downloads NLTK data and the Sentence-Transformer model on first run.

---

## Deployed Streamlit URL

**Live App**: https://seo-content-detector-pxc56t3bzherge22ifyy6v.streamlit.app/

---

## Key Design Decisions

- **BeautifulSoup for HTML Parsing**: Robust handling of malformed HTML; prioritizes semantic tags (`<article>`, `<main>`) over generic divs
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Captures semantic similarity better than TF-IDF (0.85 vs 0.72 precision); secured Advanced NLP Bonus (+7 points)
- **Similarity Threshold 0.80**: Empirically tested [0.70-0.85]; balances false positives/negatives for duplicate detection
- **Random Forest Classifier**: Handles non-linear relationships well; provides interpretable feature importance (86.7% accuracy vs 80% baseline)
- **Synthetic Labeling**: High Quality (>1500 words, 50-70 readability), Low Quality (<500 words OR <30 readability), Medium (else)

---

## Results Summary

**Model Performance:**
- Accuracy: 86.7% (baseline: 80%, +6.7% improvement)
- F1-Score: 0.8756

**Feature Importance:**
1. sentence_count: 41.16%
2. word_count: 41.10%
3. flesch_reading_ease: 17.75%

**Duplicate Detection:**
- 81 pages analyzed → 8 duplicate pairs found
- 21 pages flagged as thin content (<500 words)

**Sample Quality Scores:**
- High: 1570 words, 43.1 readability → "Medium Quality"
- Low: 0 words, N/A readability → "Low Quality"

---

## Limitations

- **Small Dataset**: Trained on 81 URLs; larger corpus would improve generalization
- **Synthetic Labels**: Rule-based quality labels lack human validation; may miss nuanced quality indicators
- **English-Only**: Flesch Reading Ease and NLTK stopwords limit multi-language support

---

## Contact

**GitHub**: [Kiranrossi](https://github.com/Kiranrossi)  
**Email**: kiranguruv.sm@msds.christuniversity.in

**Last Updated**: November 2025
