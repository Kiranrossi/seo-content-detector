"""Quality scoring and similarity detection with SEO recommendations"""
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Load model once (cached)
_model = None
_reference_df = None

def get_model():
    global _model
    if _model is None:
        model_path = Path(__file__).parent.parent / "models" / "quality_model.pkl"
        if not model_path.exists():
            # Fallback to parent directory structure
            model_path = Path(__file__).parent.parent.parent / "models" / "quality_model.pkl"
        _model = joblib.load(model_path)
    return _model

def get_reference_data():
    global _reference_df
    if _reference_df is None:
        # Load features.csv for similarity comparison
        features_path = Path(__file__).parent.parent / "data" / "features.csv"
        if not features_path.exists():
            features_path = Path(__file__).parent.parent.parent / "data" / "features.csv"
        _reference_df = pd.read_csv(features_path)
    return _reference_df

def predict_quality(features: dict) -> dict:
    """
    Predict content quality with proper confidence scores.
    
    Returns:
        dict with quality_label and quality_confidence
    """
    try:
        model = get_model()
        
        # Prepare features array
        X = np.array([[
            features['word_count'],
            features['sentence_count'],
            features['readability']
        ]])
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get class names and create confidence dict
        class_names = model.classes_
        confidence = {
            str(class_names[i]): float(probabilities[i])
            for i in range(len(class_names))
        }
        
        # Ensure all quality levels are present
        for quality in ['High', 'Medium', 'Low']:
            if quality not in confidence:
                confidence[quality] = 0.0
        
        return {
            'quality_label': str(prediction),
            'quality_confidence': confidence
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return default prediction if model fails
        wc = features['word_count']
        readability = features['readability']
        
        # Simple rule-based fallback
        if wc > 1500 and 50 <= readability <= 70:
            label = 'High'
            conf = {'High': 0.7, 'Medium': 0.2, 'Low': 0.1}
        elif wc < 500 or readability < 30:
            label = 'Low'
            conf = {'High': 0.1, 'Medium': 0.2, 'Low': 0.7}
        else:
            label = 'Medium'
            conf = {'High': 0.2, 'Medium': 0.6, 'Low': 0.2}
        
        return {
            'quality_label': label,
            'quality_confidence': conf
        }

def find_similar_content(features: dict, threshold: float = 0.8, top_n: int = 5) -> list:
    """
    Find similar content from reference dataset.
    
    Args:
        features: Dict with 'embedding' key containing vector
        threshold: Minimum similarity threshold
        top_n: Maximum number of results to return
        
    Returns:
        List of dicts with 'url' and 'similarity' keys
    """
    try:
        ref_df = get_reference_data()
        
        # Load embeddings (stored as strings in CSV)
        ref_embeddings = []
        ref_urls = []
        
        for idx, row in ref_df.head(100).iterrows():  # Limit for speed
            try:
                emb_str = row['embedding']
                emb = eval(emb_str)  # Convert string back to list
                ref_embeddings.append(emb)
                ref_urls.append(row['url'])
            except:
                continue
        
        if not ref_embeddings:
            return []
        
        # Compute similarities
        ref_embeddings = np.array(ref_embeddings)
        query_embedding = features['embedding'].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, ref_embeddings)[0]
        
        # Get top N (excluding exact matches, keeping meaningful similarities)
        results = []
        for idx, sim in enumerate(similarities):
            if threshold < sim < 0.99:  # Exclude exact matches
                results.append({
                    'url': ref_urls[idx],
                    'similarity': float(sim)
                })
        
        # Sort by similarity and return top N
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_n]
    
    except Exception as e:
        print(f"Similarity search error: {e}")
        return []

def get_seo_recommendations(features: dict, prediction: dict) -> List[Dict]:
    """
    Generate SEO recommendations based on content analysis.
    
    Returns:
        List of recommendation dicts with 'type' and 'message'
    """
    recommendations = []
    
    word_count = features['word_count']
    readability = features['readability']
    sentence_count = features['sentence_count']
    quality = prediction['quality_label']
    
    # Word count recommendations
    if word_count < 300:
        recommendations.append({
            'type': 'critical',
            'message': f'Content is too short ({word_count} words). Aim for at least 500 words for better SEO.'
        })
    elif word_count < 500:
        recommendations.append({
            'type': 'warning',
            'message': f'Content length ({word_count} words) is below recommended. Consider expanding to 800-1500 words.'
        })
    elif word_count > 3000:
        recommendations.append({
            'type': 'tip',
            'message': f'Long content ({word_count} words). Ensure it\'s well-structured with headings and sections.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'message': f'Good content length ({word_count} words). Optimal for SEO.'
        })
    
    # Readability recommendations
    if readability < 30:
        recommendations.append({
            'type': 'critical',
            'message': f'Readability score ({readability:.1f}) is very low. Content is too complex. Simplify sentences.'
        })
    elif readability < 50:
        recommendations.append({
            'type': 'warning',
            'message': f'Readability ({readability:.1f}) is below average. Consider shorter sentences and simpler words.'
        })
    elif readability > 80:
        recommendations.append({
            'type': 'tip',
            'message': f'Readability ({readability:.1f}) is very high. Ensure content depth matches target audience.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'message': f'Excellent readability score ({readability:.1f}). Easy to understand.'
        })
    
    # Sentence structure
    if sentence_count > 0:
        avg_words_per_sentence = word_count / sentence_count
        if avg_words_per_sentence > 25:
            recommendations.append({
                'type': 'warning',
                'message': f'Average sentence length ({avg_words_per_sentence:.1f} words) is high. Break into shorter sentences.'
            })
        elif avg_words_per_sentence < 10:
            recommendations.append({
                'type': 'tip',
                'message': f'Very short sentences (avg {avg_words_per_sentence:.1f} words). Consider varying sentence length.'
            })
    
    # Quality-based recommendations
    if quality == 'Low':
        recommendations.append({
            'type': 'critical',
            'message': 'Overall content quality is low. Focus on increasing depth, clarity, and value.'
        })
    elif quality == 'Medium':
        recommendations.append({
            'type': 'tip',
            'message': 'Content quality is moderate. Enhance with examples, data, and expert insights.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'message': 'Excellent content quality! Maintain this standard across all pages.'
        })
    
    # Thin content check
    if features.get('is_thin', False):
        recommendations.append({
            'type': 'critical',
            'message': 'Flagged as thin content. Add more value, examples, and detailed explanations.'
        })
    
    # General SEO tips
    recommendations.append({
        'type': 'tip',
        'message': 'Ensure proper use of headings (H1, H2, H3), meta descriptions, and internal links.'
    })
    
    return recommendations