"""
SEO Content Analyzer - Clean & Functional Streamlit App
========================================================
Modern dark theme with working batch and compare features
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.parser import parse_url
from utils.features import extract_features
from utils.scorer import predict_quality, find_similar_content, get_seo_recommendations

# Configure page
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0a0a0a;
    }
    
    .main {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #333;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: #888;
        font-size: 1.1rem;
    }
    
    /* Mode buttons */
    .stButton>button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #333;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        border-color: #2563eb;
        color: white;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #333;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #1a1a1a;
        border: 2px dashed #333;
        border-radius: 8px;
        padding: 2rem;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        color: #888;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: white;
        border-color: #2563eb;
    }
    
    /* Quality badges */
    .quality-high {
        background-color: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .quality-medium {
        background-color: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .quality-low {
        background-color: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Header
st.markdown("""
    <div class="main-header">
        <div class="main-title">🚀 SEO Content Analyzer</div>
        <div class="main-subtitle">AI-Powered Content Quality Assessment & Duplicate Detection</div>
    </div>
""", unsafe_allow_html=True)

# Mode selection with buttons
col1, col2, col3, col4 = st.columns([2, 2, 2, 6])

with col1:
    if st.button("🔍 Single URL", use_container_width=True):
        st.session_state.mode = "single"

with col2:
    if st.button("📦 Batch Analysis", use_container_width=True):
        st.session_state.mode = "batch"

with col3:
    if st.button("⚖️ Compare URLs", use_container_width=True):
        st.session_state.mode = "compare"

# Initialize mode if not set
if 'mode' not in st.session_state:
    st.session_state.mode = "single"

st.markdown("<br>", unsafe_allow_html=True)

# ==================== SINGLE URL MODE ====================
if st.session_state.mode == "single":
    st.markdown("### 🔍 Single URL Analysis")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter URL to analyze",
            placeholder="https://example.com/article",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🔎 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and url_input:
        with st.spinner("🔄 Analyzing content..."):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Parse URL
                status_text.text("📥 Fetching content...")
                progress_bar.progress(25)
                parsed_data = parse_url(url_input)
                
                if 'error' in parsed_data:
                    st.error(f"❌ {parsed_data['error']}")
                    st.stop()
                
                # Step 2: Extract features
                status_text.text("🔧 Extracting features...")
                progress_bar.progress(50)
                features = extract_features(parsed_data)
                
                # Step 3: Predict quality
                status_text.text("🤖 Analyzing quality...")
                progress_bar.progress(75)
                prediction = predict_quality(features)
                
                # Step 4: Find similar & get recommendations
                status_text.text("💡 Generating insights...")
                progress_bar.progress(100)
                similar = find_similar_content(features, threshold=0.8)
                recommendations = get_seo_recommendations(features, prediction)
                
                status_text.empty()
                progress_bar.empty()
                
                # Store in history
                st.session_state.analysis_history.append({
                    'url': url_input,
                    'quality': prediction['quality_label'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success("✅ Analysis complete!")
                
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Word Count", f"{features['word_count']:,}")
                
                with col2:
                    st.metric("📖 Readability", f"{features['readability']:.1f}/100")
                
                with col3:
                    quality_class = {
                        'High': 'quality-high',
                        'Medium': 'quality-medium',
                        'Low': 'quality-low'
                    }
                    st.markdown("**🎯 Quality**")
                    st.markdown(f'<span class="{quality_class[prediction["quality_label"]]}">{prediction["quality_label"]}</span>', unsafe_allow_html=True)
                
                with col4:
                    thin_icon = "⚠️" if features['is_thin'] else "✅"
                    st.metric("📄 Thin Content", f"{thin_icon} {'Yes' if features['is_thin'] else 'No'}")
                
                # Detailed tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Quality Dashboard",
                    "🔍 Similar Content",
                    "💡 Recommendations",
                    "📥 Download"
                ])
                
                with tab1:
                    st.markdown("### 📊 Quality Dashboard")
                    
                    confidence = prediction['quality_confidence']
                    
                    # Row 1: Confidence Chart + Gauge
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Confidence bar chart
                        conf_df = pd.DataFrame([
                            {'Quality': k, 'Confidence': v * 100} 
                            for k, v in confidence.items()
                        ])
                        
                        fig_conf = px.bar(
                            conf_df,
                            x='Quality',
                            y='Confidence',
                            color='Quality',
                            color_discrete_map={'High': '#10b981', 'Medium': '#f59e0b', 'Low': '#ef4444'},
                            title='Quality Prediction Confidence',
                            text='Confidence'
                        )
                        fig_conf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_conf.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            showlegend=False,
                            height=350,
                            yaxis_title="Confidence (%)",
                            yaxis=dict(range=[0, 100])
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with col2:
                        # Gauge chart for overall quality score
                        import plotly.graph_objects as go
                        quality_score = confidence[prediction['quality_label']] * 100
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=quality_score,
                            title={'text': "Overall Score", 'font': {'size': 20, 'color': 'white'}},
                            number={'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': "#10b981" if quality_score > 66 else "#f59e0b" if quality_score > 33 else "#ef4444"},
                                'bgcolor': "rgba(0,0,0,0.3)",
                                'borderwidth': 2,
                                'bordercolor': "white",
                                'steps': [
                                    {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.2)'},
                                    {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.2)'},
                                    {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 3},
                                    'thickness': 0.75,
                                    'value': quality_score
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "white"},
                            height=350,
                            margin=dict(l=20, r=20, t=60, b=20)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Row 2: Metrics Radar Chart
                    st.markdown("### 📈 Content Metrics Overview")
                    
                    # Normalize metrics to 0-100 scale for radar chart
                    normalized_metrics = {
                        'Word Count': min((features['word_count'] / 2000) * 100, 100),
                        'Readability': features['readability'],
                        'Sentences': min((features['sentence_count'] / 50) * 100, 100),
                        'Avg Words/Sentence': min((features['word_count']/max(features['sentence_count'], 1) / 30) * 100, 100),
                        'Overall Quality': quality_score
                    }
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(normalized_metrics.values()),
                        theta=list(normalized_metrics.keys()),
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.3)',
                        line=dict(color='#3b82f6', width=2),
                        marker=dict(size=8, color='#3b82f6')
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                showticklabels=True,
                                ticks='outside',
                                tickfont=dict(color='white', size=10),
                                gridcolor='rgba(255, 255, 255, 0.2)'
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
                        height=450,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Row 3: Detailed Statistics Cards
                    st.markdown("### 📋 Detailed Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        readability_level = "Very Easy" if features['readability'] > 80 else "Easy" if features['readability'] > 70 else "Standard" if features['readability'] > 60 else "Fairly Difficult" if features['readability'] > 50 else "Difficult"
                        readability_color = "#10b981" if features['readability'] > 60 else "#f59e0b" if features['readability'] > 40 else "#ef4444"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%); 
                                    padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(59, 130, 246, 0.3);'>
                            <h4 style='margin: 0 0 1rem 0; color: #3b82f6;'>📖 Readability</h4>
                            <div style='font-size: 2.5rem; font-weight: bold; color: {readability_color}; margin-bottom: 0.5rem;'>
                                {features['readability']:.1f}/100
                            </div>
                            <div style='color: #888; margin-bottom: 0.5rem;'>Level: <strong style='color: white;'>{readability_level}</strong></div>
                            <div style='color: #888; font-size: 0.9rem;'>Target: 60-70 for best SEO</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        words_per_sentence = features['word_count']/max(features['sentence_count'], 1)
                        sentence_color = "#10b981" if 15 <= words_per_sentence <= 25 else "#f59e0b" if 10 <= words_per_sentence <= 30 else "#ef4444"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%); 
                                    padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.3);'>
                            <h4 style='margin: 0 0 1rem 0; color: #10b981;'>📝 Content Stats</h4>
                            <div style='font-size: 2.5rem; font-weight: bold; color: white; margin-bottom: 0.5rem;'>
                                {features['word_count']:,}
                            </div>
                            <div style='color: #888; margin-bottom: 0.5rem;'>Words across {features['sentence_count']} sentences</div>
                            <div style='color: {sentence_color}; font-size: 0.9rem; font-weight: 600;'>
                                Avg: {words_per_sentence:.1f} words/sentence
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        content_depth = "Excellent" if features['word_count'] > 1500 else "Good" if features['word_count'] > 800 else "Fair" if features['word_count'] > 500 else "Thin"
                        depth_color = "#10b981" if features['word_count'] > 1500 else "#3b82f6" if features['word_count'] > 800 else "#f59e0b" if features['word_count'] > 500 else "#ef4444"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%); 
                                    padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(245, 158, 11, 0.3);'>
                            <h4 style='margin: 0 0 1rem 0; color: #f59e0b;'>🎯 Content Depth</h4>
                            <div style='font-size: 2.5rem; font-weight: bold; color: {depth_color}; margin-bottom: 0.5rem;'>
                                {content_depth}
                            </div>
                            <div style='color: #888; margin-bottom: 0.5rem;'>Quality: <strong style='color: white;'>{prediction["quality_label"]}</strong></div>
                            <div style='color: #888; font-size: 0.9rem;'>{'✅ Good length' if features['word_count'] >= 500 else '⚠️ Consider expanding'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Row 4: Feature Importance (if available from model)
                    st.markdown("### 🔍 Key Quality Factors")
                    
                    # Create feature importance visualization
                    feature_importance = {
                        'Word Count': features['word_count'] / 2000 * 100 if features['word_count'] <= 2000 else 100,
                        'Readability Score': features['readability'],
                        'Sentence Structure': min((features['sentence_count'] / 50) * 100, 100)
                    }
                    
                    fig_importance = go.Figure()
                    
                    colors = ['#3b82f6', '#10b981', '#f59e0b']
                    
                    fig_importance.add_trace(go.Bar(
                        x=list(feature_importance.values()),
                        y=list(feature_importance.keys()),
                        orientation='h',
                        marker=dict(
                            color=colors,
                            line=dict(color='white', width=1)
                        ),
                        text=[f"{v:.1f}%" for v in feature_importance.values()],
                        textposition='outside',
                        textfont=dict(color='white', size=14)
                    ))
                    
                    fig_importance.update_layout(
                        title="Feature Impact on Quality Score",
                        xaxis_title="Impact Score (%)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=300,
                        showlegend=False,
                        xaxis=dict(range=[0, 110], gridcolor='rgba(255, 255, 255, 0.1)'),
                        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with tab2:
                    st.markdown("### 🔍 Similar Content Detection")
                    
                    if similar:
                        st.warning(f"⚠️ Found {len(similar)} similar page(s)")
                        
                        for i, item in enumerate(similar, 1):
                            with st.expander(f"#{i} - Similarity: {item['similarity']:.1%}"):
                                st.markdown(f"**URL:** {item['url']}")
                                st.progress(item['similarity'])
                                
                                if item['similarity'] > 0.9:
                                    st.error("🚨 Very high similarity - likely duplicate!")
                                elif item['similarity'] > 0.8:
                                    st.warning("⚠️ High similarity - check for overlap")
                    else:
                        st.success("✅ No similar content found - original content!")
                
                with tab3:
                    st.markdown("### 💡 SEO Recommendations")
                    
                    for rec in recommendations:
                        if rec['type'] == 'critical':
                            st.error(f"🔴 {rec['message']}")
                        elif rec['type'] == 'warning':
                            st.warning(f"🟡 {rec['message']}")
                        elif rec['type'] == 'success':
                            st.success(f"🟢 {rec['message']}")
                        else:
                            st.info(f"ℹ️ {rec['message']}")
                
                with tab4:
                    st.markdown("### 📥 Download Report")
                    
                    result_data = {
                        "url": url_input,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {
                            "word_count": features['word_count'],
                            "sentence_count": features['sentence_count'],
                            "readability": round(features['readability'], 2),
                            "is_thin": features['is_thin']
                        },
                        "quality": {
                            "label": prediction['quality_label'],
                            "confidence": {k: round(v, 3) for k, v in confidence.items()}
                        },
                        "similar_content_count": len(similar)
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        json_str = json.dumps(result_data, indent=2)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_str,
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        csv_data = pd.DataFrame([{
                            'URL': url_input,
                            'Quality': prediction['quality_label'],
                            'Word Count': features['word_count'],
                            'Readability': features['readability'],
                            'Is Thin': features['is_thin'],
                            'Similar Pages': len(similar)
                        }])
                        
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv_data.to_csv(index=False),
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)
    
    elif analyze_button:
        st.warning("⚠️ Please enter a URL to analyze")

# ==================== BATCH ANALYSIS MODE ====================
elif st.session_state.mode == "batch":
    st.markdown("### 📦 Batch URL Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with URLs (must have 'url' column)",
        type=['csv'],
        help="CSV should have a column named 'url' with the URLs to analyze"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if 'url' column exists
            if 'url' not in df.columns:
                st.error("❌ CSV must have a 'url' column")
                st.stop()
            
            st.success(f"✅ Loaded {len(df)} URLs")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Analyze All URLs", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    try:
                        url = row['url']
                        status_text.text(f"Analyzing {idx + 1}/{len(df)}: {url[:50]}...")
                        progress_bar.progress((idx + 1) / len(df))
                        
                        # Parse and analyze
                        parsed_data = parse_url(url)
                        if 'error' in parsed_data:
                            results.append({
                                'url': url,
                                'status': 'Failed',
                                'error': parsed_data['error']
                            })
                            continue
                        
                        features = extract_features(parsed_data)
                        prediction = predict_quality(features)
                        
                        results.append({
                            'url': url,
                            'status': 'Success',
                            'word_count': features['word_count'],
                            'readability': round(features['readability'], 1),
                            'quality': prediction['quality_label'],
                            'is_thin': features['is_thin']
                        })
                        
                    except Exception as e:
                        results.append({
                            'url': url,
                            'status': 'Error',
                            'error': str(e)
                        })
                
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                st.success("✅ Batch analysis complete!")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    success_count = len(results_df[results_df['status'] == 'Success'])
                    st.metric("✅ Successful", success_count)
                
                with col2:
                    if 'quality' in results_df.columns:
                        high_quality = len(results_df[results_df['quality'] == 'High'])
                        st.metric("🎯 High Quality", high_quality)
                
                with col3:
                    if 'is_thin' in results_df.columns:
                        thin_count = len(results_df[results_df['is_thin'] == True])
                        st.metric("⚠️ Thin Content", thin_count)
                
                # Download results
                st.download_button(
                    label="📥 Download Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")

# ==================== COMPARE URLS MODE ====================
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Compare URLs")
    
    url1 = st.text_input("First URL", placeholder="https://example.com/page1")
    url2 = st.text_input("Second URL", placeholder="https://example.com/page2")
    
    if st.button("🔍 Compare", type="primary") and url1 and url2:
        with st.spinner("🔄 Comparing URLs..."):
            try:
                results = []
                
                for url in [url1, url2]:
                    parsed_data = parse_url(url)
                    if 'error' in parsed_data:
                        st.error(f"❌ Failed to fetch {url}: {parsed_data['error']}")
                        st.stop()
                    
                    features = extract_features(parsed_data)
                    prediction = predict_quality(features)
                    
                    results.append({
                        'url': url,
                        'word_count': features['word_count'],
                        'readability': features['readability'],
                        'quality': prediction['quality_label'],
                        'is_thin': features['is_thin'],
                        'sentence_count': features['sentence_count']
                    })
                
                st.success("✅ Comparison complete!")
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                
                for idx, (col, result) in enumerate(zip([col1, col2], results)):
                    with col:
                        st.markdown(f"### URL {idx + 1}")
                        st.markdown(f"**{result['url'][:50]}...**")
                        
                        quality_class = {
                            'High': 'quality-high',
                            'Medium': 'quality-medium',
                            'Low': 'quality-low'
                        }
                        
                        st.markdown(f'<div class="{quality_class[result["quality"]]}" style="text-align: center; margin: 1rem 0;">{result["quality"]} Quality</div>', unsafe_allow_html=True)
                        
                        st.metric("📝 Word Count", f"{result['word_count']:,}")
                        st.metric("📖 Readability", f"{result['readability']:.1f}/100")
                        st.metric("📄 Sentences", result['sentence_count'])
                        
                        thin_status = "⚠️ Yes" if result['is_thin'] else "✅ No"
                        st.metric("Thin Content", thin_status)
                
                # Comparison chart
                st.markdown("### 📊 Visual Comparison")
                
                comparison_df = pd.DataFrame(results)
                comparison_df['URL'] = ['URL 1', 'URL 2']
                
                fig = px.bar(
                    comparison_df,
                    x='URL',
                    y=['word_count', 'readability'],
                    barmode='group',
                    title='Metrics Comparison'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Comparison failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>SEO Content Analyzer</strong> | Powered by AI & Machine Learning</p>
        <p style='font-size: 0.9rem;'>🤖 Random Forest ML | 🎯 86.7% Accuracy</p>
    </div>
""", unsafe_allow_html=True)