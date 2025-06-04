import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic
from umap import UMAP
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
from datetime import datetime
import io
import base64

# Set page config
st.set_page_config(
    page_title="Big Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold;
        color: #1f77b4; text-align: center; margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem; font-weight: bold; color: #ff7f0e;
        margin-top: 2rem; margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6; padding: 1rem;
        border-radius: 0.5rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 UGMFess X Account Analysis with BERTopic Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Konfigurasi")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Preprocessed CSV File", type=['csv'],
    help="Upload a CSV file with a 'clean_text' column containing preprocessed text data"
)

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df
    
    df = load_data(uploaded_file)
    
    # Display basic info
    st.sidebar.success(f"✅ Data loaded successfully!")
    st.sidebar.metric("Total Documents", len(df))
    
    # Check for required columns
    if 'clean_text' not in df.columns:
        st.error("❌ The uploaded file must contain a 'clean_text' column with preprocessed text data.")
        st.stop()
    
    # Show column info
    with st.sidebar.expander("📋 Dataset Info"):
        st.write("**Columns:**")
        for col in df.columns:
            st.write(f"- {col}")
        # Show sample data
        st.write("**Sample Data:**")
        st.dataframe(df.head(3))
    
    # # BERTopic Parameters
    # st.sidebar.markdown("### 🎛️ BERTopic Parameters")
    # n_neighbors = st.sidebar.slider(
    #     "UMAP Neighbors",
    #     min_value=5, max_value=50,
    #     value=15, step=5,
    #     help="Jumlah neighbors untuk UMAP dimensionality reduction"
    # )
    # n_components = st.sidebar.slider(
    #     "UMAP Components",
    #     min_value=2, max_value=10,
    #     value=5, step=1,
    #     help="Jumlah komponen untuk UMAP"
    # )

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Topic Analysis", "📊 Visualizations", "📋 Topic Details"])
    
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            # Calculate average text length
            avg_length = df['clean_text'].astype(str).str.len().mean()
            st.metric("Avg Text Length", f"{avg_length:.0f} chars")
        with col3:
            # Count non-empty texts
            non_empty = df['clean_text'].astype(str).str.strip().ne('').sum()
            st.metric("Non-empty Texts", non_empty)
        
        # Word Cloud
        st.markdown('<div class="section-header">Word Cloud</div>', unsafe_allow_html=True)
        
        if st.button("🎨 Generate Word Cloud"):
            with st.spinner("Generating word cloud..."):
                # Prepare text data
                text_data = " ".join(df['clean_text'].astype(str).fillna(''))
                
                if text_data.strip():
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        max_words=100, min_font_size=10
                    ).generate(text_data)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No text data available for word cloud generation.")
    
    with tab2:
        st.markdown('<div class="section-header">BERTopic Analysis</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Topic Modeling"):
            with st.spinner("Running BERTopic analysis... This may take a few minutes."):
                # Prepare data
                df_clean = df.copy()
                df_clean['clean_text'] = df_clean['clean_text'].fillna('')
                
                # Filter out empty texts
                df_clean = df_clean[df_clean['clean_text'].str.strip() != '']
                
                if len(df_clean) == 0:
                    st.error("❌ No valid text data found after cleaning.")
                    st.stop()
                
                # Initialize models
                sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                cluster_model = HDBSCAN(min_cluster_size=25, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
                umap_model = UMAP(
                    n_neighbors=15, n_components=5,
                    min_dist=0.0, metric='cosine', random_state=42
                )
                ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
                
                # Create BERTopic model
                topic_model = BERTopic(
                    language = "multilingual", embedding_model=sentence_model, umap_model=umap_model,
                    hdbscan_model=cluster_model, ctfidf_model=ctfidf_model, verbose=True
                )
                
                # Fit the model
                topics, probs = topic_model.fit_transform(df_clean['clean_text'].tolist())
                
                # Store results in session state
                st.session_state.topic_model = topic_model
                st.session_state.topics = topics
                st.session_state.probs = probs
                st.session_state.df_analyzed = df_clean
                
                st.success("✅ Topic modeling completed successfully!")
        
        # Display results if available
        if hasattr(st.session_state, 'topic_model'):
            topic_info = st.session_state.topic_model.get_topic_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Topics Found", len(topic_info))
            with col2:
                st.metric("Documents Analyzed", len(st.session_state.topics))
            
            # Topic info table
            st.markdown("### 📋 Topic Information")
            st.dataframe(topic_info.head(10), use_container_width=True)

            # Topic info table
            st.markdown("### 📈 Cluster Stability")
            # Ambil nilai stabilitas dan label dari HDBSCAN
            topic_model = st.session_state.topic_model
            cluster_persistence = topic_model.hdbscan_model.cluster_persistence_
            labels = topic_model.hdbscan_model.labels_
            
            # Hanya ambil topik unik (tanpa noise -1)
            valid_indices = labels != -1
            unique_topics = sorted(list(set(labels[valid_indices])))
            
            # Ambil stabilitas hanya untuk topik unik (HDBSCAN menyimpan hanya untuk klaster valid)
            stability_per_topic = cluster_persistence[:len(unique_topics)]
            
            # Buat DataFrame
            cluster_stability_df = pd.DataFrame({
                "Topic": unique_topics,
                "Stability": stability_per_topic
            }).sort_values(by="Stability", ascending=False)
            
            # Tampilkan 10 stabilitas teratas
            st.markdown("### 📈 Top 10 Cluster Stability")
            st.dataframe(cluster_stability_df.head(10), use_container_width=True)
            
            # Opsional: Visualisasi bar chart untuk top 10
            st.markdown("### 📊 Stability Bar Chart (Top 10)")
            st.bar_chart(cluster_stability_df.head(10).set_index("Topic"))
        
    with tab3:
        st.markdown('<div class="section-header">Topic Visualizations</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'topic_model'):
            topic_model = st.session_state.topic_model
            
            # Visualization options
            viz_option = st.selectbox(
                "Select Visualization",
                [
                    "Word Cloud",
                    "Topic Bar Chart",
                    "Topic Probability Distribution",
                    "Topic Hierarchy",
                    "Topic Similarity Heatmap",
                    "Topic Terms Rank",
                    "Intertopic Distance Map",
                ]
            )
            
            try:
                if viz_option == "Word Cloud":
                    st.markdown("### 🌐 Word Cloud for Topics")
                    # Choose a specific topic for word cloud
                    topic_id = st.selectbox(
                        "Select Topic for Word Cloud",
                        options=topic_model.get_topic_info()['Topic'].tolist(),
                        format_func=lambda x: f"Topic {x}: {topic_model.get_topic(x)[0][0] if x in topic_model.get_topic_info()['Topic'].tolist() else 'Unknown'}"
                    )
                    fig = topic_model.visualize_wordcloud(topic_id)
                    st.plotly_chart(fig, use_container_width=True)
                elif viz_option == "Topic Bar Chart":
                    st.markdown("### 📊 Top 10 Topic Bar Chart")
                    fig = topic_model.visualize_barchart(top_n_topics=10)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif viz_option == "Topic Probability Distribution":
                    st.markdown("### 🖥️ Topic Probability Distribution")
    
                    # Ambil probs dari session_state
                    if hasattr(st.session_state, 'probs'):
                        probs = st.session_state.probs
                        fig = topic_model.visualize_distribution(probs)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Probability data not available. Please run topic modeling first.")

                    
                elif viz_option == "Topic Hierarchy":
                    st.markdown("### 🌳 Topic Hierarchy")
                    fig = topic_model.visualize_hierarchy()
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Topic Similarity Heatmap":
                    st.markdown("### 🔥 Topic Similarity Heatmap")
                    fig = topic_model.visualize_heatmap()
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Topic Terms Rank":
                    st.markdown("### 📈 Topic Terms Rank")
                    fig = topic_model.visualize_term_rank()
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Intertopic Distance Map":
                    st.markdown("### 🗺️ Intertopic Distance Map")
                    fig = topic_model.visualize_topics()
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error generating visualization: {str(e)}")
        else:
            st.info("ℹ️ Please run topic modeling first in the Topic Analysis tab.")
    
    with tab4:
        st.markdown('<div class="section-header">Detailed Topic Analysis</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'topic_model'):
            topic_model = st.session_state.topic_model
            topic_info = topic_model.get_topic_info()
            
            # Topic selection
            topic_ids = topic_info['Topic'].tolist()
            selected_topic = st.selectbox(
                "Select Topic for Detailed Analysis",
                topic_ids,
                format_func=lambda x: f"Topic {x}: {topic_info[topic_info['Topic']==x]['Name'].iloc[0] if x in topic_ids else 'Unknown'}"
            )
            
            if selected_topic is not None:
                # Get topic details
                topic_words = topic_model.get_topic(selected_topic)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🏷️ Topic Keywords")
                    if topic_words:
                        keywords_df = pd.DataFrame(topic_words, columns=['Word', 'Score'])
                        st.dataframe(keywords_df, use_container_width=True)
                    else:
                        st.info("No keywords available for this topic.")
                
                with col2:
                    st.markdown("### 📊 Topic Statistics")
                    topic_row = topic_info[topic_info['Topic'] == selected_topic].iloc[0]
                    
                    st.metric("Document Count", topic_row['Count'])
                    st.metric("Topic Name", topic_row['Name'])
                    
                    # Show representative documents
                    if hasattr(st.session_state, 'df_analyzed') and hasattr(st.session_state, 'topics'):
                        topic_docs = st.session_state.df_analyzed[
                            np.array(st.session_state.topics) == selected_topic
                        ]['clean_text'].head(5)
                        
                        st.markdown("### 📄 Sample Documents")
                        for i, doc in enumerate(topic_docs, 1):
                            with st.expander(f"Document {i}"):
                                st.write(doc[:500] + "..." if len(doc) > 500 else doc)
            
            # Download results
            st.markdown("### 💾 Export Results")
            
            if st.button("📥 Download Topic Results"):
                # Prepare results for download
                results_df = topic_info.copy()
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Topics CSV", data=csv_data,
                    file_name="topic_analysis_results.csv", mime="text/csv"
                )
        else:
            st.info("ℹ️ Please run topic modeling first in the Topic Analysis tab.")

else:
    # Instructions when no file is uploaded
    st.markdown("""
    ## 🚀 Big Data Analysis Dashboard
    
    Analisis post UGMFess X menggunakan BERTopic untuk menemukan topik-topik menarik dalam rentang waktu Juni 2024 hingga saat ini.
    Dashboard ini memungkinkan Anda untuk melakukan analisis topik secara interaktif dengan berbagai visualisasi dan analisis mendalam.
    
    ### 📋 Instruksi:
    1. **Unggah file CSV yang telah dilakukan preprocessing** menggunakan pengunggah file di sidebar
    2. **Pastikan CSV mengandung kolom 'clean_text'** dengan data teks yang telah diproses
    3. **Konfigurasi parameter BERTopic** di sidebar (opsional)
    4. **Jelajahi tab** untuk mengeksplorasi berbagai aspek data Anda:
       - **Overview**: Statistik dataset dan word cloud
       - **Topic Analysis**: Jalankan pemodelan BERTopic
       - **Visualizations**: Visualisasi topik interaktif
       - **Topic Details**: Analisis mendalam topik individu

    **Selamat Mencoba** 🎉
                
    **Disusun oleh**:
    1. **Krismantoro Bagus Meidianto**      (22/492709/PA/21137)
    2. **Nicolas Dwi Hardjoleksono**		(22/493899/PA/21225)
    3. **Azhar Bagaskara**			        (22/502652/PA/21573)
    4. **Muhammad Haikal Syafi Alawiy**	    (22/503880/PA/21669)
    5. **Hafizh Al Muzakar**			    (22/505360/PA/21760)

    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Built with ❤️ using Streamlit and BERTopic
    </div>
    """,
    unsafe_allow_html=True
)
