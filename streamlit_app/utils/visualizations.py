"""Enhanced visualization utilities for SEO dashboard"""
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

def create_quality_gauge(quality_score: float, quality_label: str) -> go.Figure:
    """
    Create an enhanced gauge chart for quality score.
    
    Args:
        quality_score: Score from 0-100
        quality_label: Quality label (High/Medium/Low)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=quality_score,
        title={'text': "Overall Quality Score", 'font': {'size': 24, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 50, 'color': 'white'}},
        delta={'reference': 70, 'increasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {
                'color': "#10b981" if quality_score > 66 else "#f59e0b" if quality_score > 33 else "#ef4444"
            },
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 3,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': quality_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def create_confidence_chart(confidence: Dict[str, float]) -> go.Figure:
    """
    Create enhanced confidence distribution bar chart.
    
    Args:
        confidence: Dict with quality labels and confidence scores
        
    Returns:
        Plotly figure object
    """
    labels = list(confidence.keys())
    values = [v * 100 for v in confidence.values()]
    
    colors = {
        'High': '#10b981',
        'Medium': '#f59e0b',
        'Low': '#ef4444'
    }
    
    bar_colors = [colors.get(label, '#888') for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=bar_colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            textfont=dict(size=16, color='white'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Quality Prediction Confidence",
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Quality Level",
        yaxis_title="Confidence (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        showlegend=False,
        height=400,
        yaxis=dict(
            range=[0, 110],
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=12)
        ),
        xaxis=dict(tickfont=dict(size=14))
    )
    
    return fig


def create_metrics_radar(features: dict, quality_score: float) -> go.Figure:
    """
    Create radar chart for content metrics.
    
    Args:
        features: Dict with content features
        quality_score: Overall quality score
        
    Returns:
        Plotly figure object
    """
    # Normalize all metrics to 0-100 scale
    word_count_normalized = min((features['word_count'] / 2000) * 100, 100)
    readability_normalized = features['readability']
    sentence_count_normalized = min((features['sentence_count'] / 50) * 100, 100)
    
    avg_words_per_sentence = features['word_count'] / max(features['sentence_count'], 1)
    sentence_structure_normalized = min((avg_words_per_sentence / 30) * 100, 100)
    
    categories = [
        'Word Count<br>(0-2000+)',
        'Readability<br>(0-100)',
        'Sentences<br>(0-50+)',
        'Sentence Structure<br>(Words/Sent)',
        'Overall Quality'
    ]
    
    values = [
        word_count_normalized,
        readability_normalized,
        sentence_count_normalized,
        sentence_structure_normalized,
        quality_score
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color='#3b82f6', symbol='circle'),
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='outside',
                tickfont=dict(color='white', size=12),
                gridcolor='rgba(255, 255, 255, 0.2)',
                tickvals=[0, 25, 50, 75, 100]
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='white', size=12)
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=500,
        showlegend=False,
        title={
            'text': "Content Metrics Overview",
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95
        }
    )
    
    return fig


def create_feature_importance_chart(features: dict) -> go.Figure:
    """
    Create horizontal bar chart showing feature importance.
    
    Args:
        features: Dict with content features
        
    Returns:
        Plotly figure object
    """
    feature_scores = {
        'Word Count': min((features['word_count'] / 2000) * 100, 100),
        'Readability': features['readability'],
        'Sentence Count': min((features['sentence_count'] / 50) * 100, 100),
        'Sentence Structure': min(((features['word_count'] / max(features['sentence_count'], 1)) / 30) * 100, 100)
    }
    
    # Sort by value
    sorted_features = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(sorted_features.values()),
        y=list(sorted_features.keys()),
        orientation='h',
        marker=dict(
            color=colors[:len(sorted_features)],
            line=dict(color='white', width=2)
        ),
        text=[f"{v:.1f}%" for v in sorted_features.values()],
        textposition='outside',
        textfont=dict(color='white', size=14),
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Feature Impact Analysis",
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Impact Score (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        height=350,
        showlegend=False,
        xaxis=dict(
            range=[0, 110],
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        margin=dict(l=150, r=50, t=80, b=50)
    )
    
    return fig


def create_comparison_chart(results: List[Dict]) -> go.Figure:
    """
    Create side-by-side comparison chart for multiple URLs.
    
    Args:
        results: List of analysis result dicts
        
    Returns:
        Plotly figure object
    """
    if not results or len(results) < 2:
        return None
    
    urls = [f"URL {i+1}" for i in range(len(results))]
    
    fig = go.Figure()
    
    # Word count comparison
    fig.add_trace(go.Bar(
        name='Word Count (÷10)',
        x=urls,
        y=[r['word_count'] / 10 for r in results],
        marker_color='#3b82f6',
        hovertemplate='<b>Word Count</b><br>%{y:.0f} words<extra></extra>'
    ))
    
    # Readability comparison
    fig.add_trace(go.Bar(
        name='Readability',
        x=urls,
        y=[r['readability'] for r in results],
        marker_color='#10b981',
        hovertemplate='<b>Readability</b><br>%{y:.1f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "URL Metrics Comparison",
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='URLs',
        yaxis_title='Score',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'size': 14},
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
    )
    
    return fig


def create_word_cloud(text: str) -> plt.Figure:
    """
    Generate word cloud from text content.
    
    Args:
        text: Input text string
        
    Returns:
        matplotlib Figure object
    """
    if not text or len(text.strip()) == 0:
        # Return empty figure if no text
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No text available', 
                ha='center', va='center', fontsize=20, color='white')
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')
        fig.patch.set_facecolor('#1a1a1a')
        return fig
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='#1a1a1a',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=12,
        collocations=False
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout(pad=0)
    
    return fig


def create_trend_chart(history: List[Dict]) -> go.Figure:
    """
    Create trend chart from analysis history.
    
    Args:
        history: List of historical analysis results
        
    Returns:
        Plotly figure showing trends
    """
    if not history or len(history) == 0:
        return None
    
    timestamps = [h['timestamp'] for h in history]
    qualities = [h['quality'] for h in history]
    
    # Convert quality to numeric
    quality_map = {'High': 3, 'Medium': 2, 'Low': 1}
    quality_scores = [quality_map.get(q, 0) for q in qualities]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=quality_scores,
        mode='lines+markers',
        name='Quality Trend',
        line=dict(color='#3b82f6', width=3),
        marker=dict(
            size=12,
            color=quality_scores,
            colorscale=['#ef4444', '#f59e0b', '#10b981'],
            showscale=False,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Quality: %{text}<extra></extra>',
        text=qualities
    ))
    
    fig.update_layout(
        title={
            'text': 'Content Quality Trend Over Time',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Timestamp',
        yaxis_title='Quality Level',
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High'],
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'size': 14},
        height=400
    )
    
    return fig


def create_dashboard_charts(features: dict, prediction: dict) -> Dict:
    """
    Create comprehensive dashboard visualizations.
    
    Returns:
        Dict of plotly figure objects
    """
    quality_score = prediction['quality_confidence'][prediction['quality_label']] * 100
    
    charts = {
        'gauge': create_quality_gauge(quality_score, prediction['quality_label']),
        'confidence': create_confidence_chart(prediction['quality_confidence']),
        'radar': create_metrics_radar(features, quality_score),
        'importance': create_feature_importance_chart(features)
    }
    
    return charts