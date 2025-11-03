"""Feature extraction utilities with proper word count calculation"""
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
from sentence_transformers import SentenceTransformer
import re

# Load model once (cached)
_model = None

def get_embedding_model():
    """Get or load the sentence transformer model"""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
    
    return text.strip()

def calculate_word_count(text: str) -> int:
    """
    Calculate accurate word count.
    
    Uses multiple methods to ensure accuracy:
    1. Simple split on whitespace
    2. NLTK word tokenization (more accurate)
    
    Args:
        text: Input text
        
    Returns:
        Integer word count
    """
    if not text or len(text.strip()) == 0:
        return 0
    
    try:
        # Method 1: NLTK tokenization (more accurate)
        tokens = word_tokenize(text)
        # Filter out pure punctuation
        words = [w for w in tokens if any(c.isalnum() for c in w)]
        return len(words)
    except:
        # Fallback: simple split
        return len(text.split())

def extract_features(parsed_data: dict) -> dict:
    """
    Extract comprehensive NLP features from parsed content.
    
    Features extracted:
        - word_count: Total number of words (FIXED - now accurate)
        - sentence_count: Number of sentences
        - readability: Flesch Reading Ease score (0-100)
        - is_thin: Boolean flag for thin content (<500 words)
        - embedding: 384-dim sentence embedding vector
        - title: Page title
        - url: Source URL
    
    Args:
        parsed_data: Dict with 'body_text', 'title', 'url' keys
        
    Returns:
        Dict with extracted features
    """
    text = parsed_data.get('body_text', '')
    
    # Handle empty text
    if not text or len(text.strip()) == 0:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'readability': 0.0,
            'is_thin': True,
            'embedding': None,
            'title': parsed_data.get('title', ''),
            'url': parsed_data.get('url', '')
        }
    
    # Clean text first
    text_clean = clean_text(text)
    
    # Basic metrics - FIXED WORD COUNT CALCULATION
    word_count = calculate_word_count(text_clean)
    
    # Sentence count
    try:
        sentences = sent_tokenize(text_clean)
        sentence_count = len(sentences)
    except:
        # Fallback: count periods, exclamation, question marks
        sentence_count = len(re.findall(r'[.!?]+', text_clean))
        if sentence_count == 0:
            sentence_count = 1
    
    # Readability score
    try:
        if word_count > 0 and sentence_count > 0:
            readability = textstat.flesch_reading_ease(text_clean)
        else:
            readability = 0.0
    except Exception as e:
        print(f"Readability calculation error: {e}")
        readability = 0.0
    
    # Ensure readability is within valid range
    readability = max(0.0, min(100.0, readability))
    
    # Thin content flag (less than 500 words)
    is_thin = word_count < 500
    
    # Generate embedding
    try:
        model = get_embedding_model()
        # Use original text for embedding (not cleaned)
        embedding = model.encode([text])[0]
    except Exception as e:
        print(f"Embedding generation error: {e}")
        # Return zero vector if embedding fails
        import numpy as np
        embedding = np.zeros(384)
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'readability': round(readability, 2),
        'is_thin': is_thin,
        'embedding': embedding,
        'title': parsed_data.get('title', ''),
        'url': parsed_data.get('url', '')
    }

def extract_keywords_tfidf(text: str, top_n: int = 5) -> list:
    """
    Extract top keywords using simple word frequency.
    
    Args:
        text: Input text
        top_n: Number of keywords to return
        
    Returns:
        List of top keywords
    """
    if not text:
        return []
    
    try:
        from nltk.corpus import stopwords
        import nltk
        
        # Ensure stopwords are downloaded
        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        # Tokenize and filter
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(words)
        
        # Get top N
        top_keywords = [word for word, count in word_freq.most_common(top_n)]
        
        return top_keywords
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []

def get_content_statistics(features: dict) -> dict:
    """
    Generate additional content statistics.
    
    Args:
        features: Dict with basic features
        
    Returns:
        Dict with additional statistics
    """
    stats = {}
    
    # Average words per sentence
    if features['sentence_count'] > 0:
        stats['avg_words_per_sentence'] = features['word_count'] / features['sentence_count']
    else:
        stats['avg_words_per_sentence'] = 0
    
    # Readability level
    readability = features['readability']
    if readability >= 90:
        stats['readability_level'] = 'Very Easy'
    elif readability >= 70:
        stats['readability_level'] = 'Easy'
    elif readability >= 60:
        stats['readability_level'] = 'Standard'
    elif readability >= 50:
        stats['readability_level'] = 'Fairly Difficult'
    elif readability >= 30:
        stats['readability_level'] = 'Difficult'
    else:
        stats['readability_level'] = 'Very Difficult'
    
    # Content quality estimate
    word_count = features['word_count']
    if word_count > 1500 and 50 <= readability <= 70:
        stats['estimated_quality'] = 'High'
    elif word_count < 500 or readability < 30:
        stats['estimated_quality'] = 'Low'
    else:
        stats['estimated_quality'] = 'Medium'
    
    return stats