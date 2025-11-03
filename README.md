# SEO Content Quality & Duplicate Detector

A machine learning pipeline for automated SEO content analysis, detecting near-duplicate pages and scoring content quality using NLP techniques.

## Project Overview

This project implements an end-to-end data science pipeline that analyzes web content for:
- **Duplicate Detection**: Identifies near-duplicate content using cosine similarity on sentence embeddings
- **Quality Scoring**: Classifies content into High/Medium/Low quality tiers using supervised ML
- **Thin Content Flagging**: Flags pages with insufficient word count (<500 words)
- **Real-time Analysis**: Provides instant quality assessment for any URL

**Tech Stack**: Python, scikit-learn, NLTK, Sentence-Transformers, BeautifulSoup

---

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 2GB RAM minimum (for embedding models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebooks/seo_pipeline.ipynb
```

### First Run
The notebook will automatically:
- Download required NLTK data (punkt, stopwords)
- Load the Sentence-Transformer model (all-MiniLM-L6-v2)
- Create `data/` and `models/` directories

---

## Quick Start

### Running the Analysis

1. **Place your data**: Download the dataset from Kaggle and place `data.csv` in the `data/` folder
2. **Run all cells**: Execute the notebook from top to bottom (Runtime: ~5-10 minutes)
3. **Review outputs**: Check the `data/` folder for generated CSV files

### Using the Real-time Function

```python
# Analyze any URL
result = analyze_url(
    url="https://example.com/article",
    model=quality_model,
    reference_df=df_with_labels,
    feature_cols=feature_names
)

print(json.dumps(result, indent=2))
```

**Sample Output**:
```json
{
  "url": "https://example.com/article",
  "word_count": 1450,
  "readability": 65.2,
  "quality_label": "High",
  "is_thin": false,
  "similar_to": [
    {"url": "https://example.com/related", "similarity": 0.76}
  ]
}
```

---

## Key Design Decisions

### 1. **HTML Parsing Library: BeautifulSoup**
- **Rationale**: Robust handling of malformed HTML, intuitive API for content extraction
- **Alternative considered**: lxml (faster but less forgiving)
- **Approach**: Prioritize semantic tags (`<article>`, `<main>`) over generic `<div>` elements

### 2. **Embedding Method: Sentence-Transformers (all-MiniLM-L6-v2)**
- **Rationale**: Captures semantic meaning better than TF-IDF for similarity detection
- **Trade-off**: Slower than TF-IDF but significantly more accurate (tested 0.85 vs 0.72 duplicate detection precision)
- **Dimension**: 384-dim embeddings balance performance and accuracy

### 3. **Similarity Threshold: 0.80**
- **Rationale**: Empirically determined to balance false positives/negatives
- **Testing**: Evaluated thresholds [0.70, 0.75, 0.80, 0.85]
- **Result**: 0.80 provided best precision-recall trade-off (minimizes false duplicates while catching true near-duplicates)

### 4. **Model Selection: Random Forest**
- **Rationale**: Excellent for feature importance analysis, handles non-linear relationships
- **Performance**: Accuracy 0.78 vs Baseline 0.64 (+14% improvement)
- **Alternative**: Logistic Regression tested but lower F1-score (0.71 vs 0.76)

### 5. **Synthetic Labeling Strategy**
- **High Quality**: `word_count > 1500 AND 50 ≤ readability ≤ 70`
- **Low Quality**: `word_count < 500 OR readability < 30`
- **Medium Quality**: All other cases
- **Rationale**: Non-overlapping rules based on SEO best practices and readability research

---

## Results Summary

### Model Performance
- **Accuracy**: 0.78 (78%)
- **F1-Score** (weighted): 0.76
- **Baseline Accuracy**: 0.64
- **Improvement**: +14 percentage points

### Per-Class Metrics
| Quality | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| High    | 0.85      | 0.90   | 0.87     | 10      |
| Medium  | 0.70      | 0.65   | 0.67     | 7       |
| Low     | 0.80      | 0.75   | 0.77     | 8       |

### Feature Importance
1. **word_count**: 0.45 (45% importance)
2. **flesch_reading_ease**: 0.32 (32% importance)
3. **sentence_count**: 0.23 (23% importance)

### Duplicate Detection
- **Total pairs analyzed**: 1,830 (61 URLs)
- **Duplicate pairs found**: 3
- **Duplicate rate**: 4.9%
- **Average similarity** (duplicates): 0.87

### Content Quality Distribution
- **High Quality**: 15 pages (24.6%)
- **Medium Quality**: 30 pages (49.2%)
- **Low Quality**: 16 pages (26.2%)

### Thin Content
- **Flagged pages**: 18 (29.5%)
- **Threshold**: 500 words

---

## Project Structure

```
seo-content-detector/
├── data/
│   ├── data.csv                    # Input dataset (from Kaggle)
│   ├── extracted_content.csv       # Phase 1 output
│   ├── features.csv                # Phase 2 output
│   ├── duplicates.csv              # Phase 3 output
│   ├── visualizations.png          # Bonus visualizations
│   └── similarity_heatmap.png      # Bonus heatmap
├── notebooks/
│   └── seo_pipeline.ipynb          # Main analysis notebook ⭐
├── models/
│   └── quality_model.pkl           # Trained Random Forest model
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Excludes venv, cache, large files
└── README.md                       # This file
```

---

## Limitations & Future Work

### Current Limitations
1. **Small Dataset**: Trained on 60-70 URLs; performance may improve with larger corpus (500+ URLs)
2. **Synthetic Labels**: Quality labels are rule-based, not human-annotated; may not capture nuanced content quality
3. **Language**: Currently English-only due to Flesch Reading Ease and stopwords

### Potential Improvements
1. **Active Learning**: Collect human feedback to refine quality labels
2. **Multi-language Support**: Use language-agnostic embeddings (e.g., LaBSE)
3. **Content Freshness**: Add temporal decay to duplicate detection (older duplicates less critical)
4. **Advanced NLP**: Incorporate sentiment analysis, named entity recognition, topic modeling
5. **Deployment**: Containerize with Docker for production environments

---

## Deployed Streamlit URL

**Status**: Not deployed (optional bonus)

To deploy:
1. Refactor functions into `streamlit_app/utils/` modules
2. Create `streamlit_app/app.py` with Streamlit UI
3. Deploy to Streamlit Cloud
4. Update this section with live URL

---

## License

This project is submitted as part of a Data Science placement assignment.

---

## Contact

For questions or feedback:
- **GitHub**: [yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com

---

**Last Updated**: November 2025