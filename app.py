import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os
import tempfile
import json
from datetime import datetime
import pickle

# Database
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Document parsing
import PyPDF2
import docx
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
import markdown

# NLP and topic modeling
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel

# Clustering and similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
Base = declarative_base()

class AnalysisSession(Base):
    __tablename__ = 'analysis_sessions'
    
    id = Column(Integer, primary_key=True)
    session_name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_documents = Column(Integer)
    num_topics = Column(Integer)
    clustering_algorithm = Column(String(50))
    parameters = Column(Text)  # JSON string of all parameters
    document_names = Column(Text)  # JSON array
    documents_data = Column(LargeBinary)  # Pickled documents
    topic_distributions = Column(LargeBinary)  # Pickled topic distributions
    topics_data = Column(LargeBinary)  # Pickled topics
    cluster_labels = Column(LargeBinary)  # Pickled cluster labels
    similarity_matrix = Column(LargeBinary)  # Pickled similarity matrix

@st.cache_resource
def get_database_engine():
    """Get SQLAlchemy engine with connection pooling"""
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)
        return engine
    return None

def save_analysis_session(session_name, documents, doc_names, topic_distributions, topics, cluster_labels, similarity_matrix, parameters):
    """Save analysis session to database"""
    engine = get_database_engine()
    if not engine:
        return False
    
    Session = sessionmaker(bind=engine)
    db_session = Session()
    
    try:
        analysis_session = AnalysisSession(
            session_name=session_name,
            created_at=datetime.utcnow(),
            num_documents=len(documents),
            num_topics=parameters['num_topics'],
            clustering_algorithm=parameters['clustering_algorithm'],
            parameters=json.dumps(parameters),
            document_names=json.dumps(doc_names),
            documents_data=pickle.dumps(documents),
            topic_distributions=pickle.dumps(topic_distributions),
            topics_data=pickle.dumps(topics),
            cluster_labels=pickle.dumps(cluster_labels),
            similarity_matrix=pickle.dumps(similarity_matrix)
        )
        
        db_session.add(analysis_session)
        db_session.commit()
        return True
    except Exception as e:
        db_session.rollback()
        st.error(f"Failed to save session: {str(e)}")
        return False
    finally:
        db_session.close()

def load_analysis_sessions():
    """Load all analysis sessions from database"""
    engine = get_database_engine()
    if not engine:
        return []
    
    Session = sessionmaker(bind=engine)
    db_session = Session()
    
    try:
        sessions = db_session.query(AnalysisSession).order_by(AnalysisSession.created_at.desc()).all()
        return sessions
    except Exception as e:
        st.error(f"Failed to load sessions: {str(e)}")
        return []
    finally:
        db_session.close()

def load_session_by_id(session_id):
    """Load a specific analysis session by ID"""
    engine = get_database_engine()
    if not engine:
        return None
    
    Session = sessionmaker(bind=engine)
    db_session = Session()
    
    try:
        session = db_session.query(AnalysisSession).filter_by(id=session_id).first()
        return session
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
        return None
    finally:
        db_session.close()

# Document parsing functions
def extract_text_from_txt(file):
    """Extract text from .txt file"""
    return file.read().decode('utf-8', errors='ignore')

def extract_text_from_docx(file):
    """Extract text from .docx file"""
    doc = docx.Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def extract_text_from_pdf(file):
    """Extract text from .pdf file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return '\n'.join(text)

def extract_text_from_rtf(file):
    """Extract text from .rtf file"""
    rtf_content = file.read().decode('utf-8', errors='ignore')
    return rtf_to_text(rtf_content)

def extract_text_from_html(file):
    """Extract text from .html file"""
    html_content = file.read().decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

def extract_text_from_markdown(file):
    """Extract text from .md file"""
    md_content = file.read().decode('utf-8', errors='ignore')
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

def parse_document(file):
    """Parse document based on file type"""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return extract_text_from_txt(file)
    elif file_extension == 'docx':
        return extract_text_from_docx(file)
    elif file_extension == 'doc':
        st.error(f"Legacy .doc format is not supported. Please convert '{file.name}' to .docx format.")
        return None
    elif file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'rtf':
        return extract_text_from_rtf(file)
    elif file_extension in ['html', 'htm']:
        return extract_text_from_html(file)
    elif file_extension == 'md':
        return extract_text_from_markdown(file)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

# Text preprocessing
def preprocess_text(text):
    """Preprocess text: tokenize, lowercase, remove stopwords"""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
    
    return tokens

# LDA Topic Modeling
def perform_lda(documents, num_topics=5, num_words=10, passes=10, iterations=50, alpha='auto', eta='auto'):
    """Perform LDA topic modeling on documents"""
    # Preprocess all documents
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    
    # Validate that we have a non-empty vocabulary
    if len(dictionary) == 0:
        raise ValueError("No valid words found in documents after preprocessing. Documents may be too short or contain only stopwords.")
    
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Check if corpus has content
    if all(len(doc) == 0 for doc in corpus):
        raise ValueError("All documents resulted in empty vocabularies after preprocessing. Please upload documents with more substantive content.")
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        iterations=iterations,
        alpha=alpha,
        eta=eta,
        per_word_topics=True
    )
    
    # Get topic distributions for each document
    topic_distributions = []
    for doc_bow in corpus:
        topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        topic_dist = sorted(topic_dist, key=lambda x: x[0])
        topic_distributions.append([prob for _, prob in topic_dist])
    
    # Get topics with words
    topics = []
    for idx in range(num_topics):
        topic_words = lda_model.show_topic(idx, num_words)
        topics.append(topic_words)
    
    return lda_model, topic_distributions, topics, dictionary, corpus

# Similarity computation
def compute_similarity_matrix(topic_distributions):
    """Compute cosine similarity matrix from topic distributions"""
    topic_matrix = np.array(topic_distributions)
    similarity_matrix = cosine_similarity(topic_matrix)
    return similarity_matrix

# Clustering
def perform_clustering(topic_distributions, algorithm='KMeans', num_clusters=3, eps=0.5, min_samples=2):
    """Perform clustering on topic distributions"""
    topic_matrix = np.array(topic_distributions)
    
    if algorithm == 'KMeans':
        clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(topic_matrix)
    elif algorithm == 'DBSCAN':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clusterer.fit_predict(topic_matrix)
    elif algorithm == 'Hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(topic_matrix)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    return cluster_labels, clusterer

# Dimensionality reduction for visualization
def reduce_dimensions(topic_distributions, method='PCA'):
    """Reduce topic distributions to 2D using PCA or t-SNE"""
    topic_matrix = np.array(topic_distributions)
    
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(topic_matrix)-1))
    
    reduced_data = reducer.fit_transform(topic_matrix)
    return reduced_data

# Visualization functions
def plot_similarity_heatmap(similarity_matrix, doc_names):
    """Create interactive similarity heatmap using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=doc_names,
        y=doc_names,
        colorscale='Viridis',
        text=np.round(similarity_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title="Document Similarity Heatmap",
        xaxis_title="Documents",
        yaxis_title="Documents",
        height=600,
        width=800
    )
    
    return fig

def plot_clusters(reduced_data, cluster_labels, doc_names, method='PCA'):
    """Plot document clusters in 2D space"""
    df = pd.DataFrame({
        'x': reduced_data[:, 0],
        'y': reduced_data[:, 1],
        'Cluster': [f'Cluster {label}' for label in cluster_labels],
        'Document': doc_names
    })
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Cluster',
        text='Document',
        title=f'Document Clusters ({method})',
        labels={'x': f'{method} Component 1', 'y': f'{method} Component 2'},
        height=600,
        width=800
    )
    
    fig.update_traces(textposition='top center', marker=dict(size=12))
    
    return fig

def create_wordcloud(topic_words, topic_num):
    """Create word cloud for a topic"""
    # Create word frequency dictionary
    word_freq = {word: weight for word, weight in topic_words}
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Topic {topic_num} Word Cloud', fontsize=16, fontweight='bold')
    
    return fig

# Download functions
def get_csv_download_link(df, filename):
    """Generate download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

# Main Streamlit App
def main():
    st.set_page_config(page_title="Document Similarity & Clustering Dashboard", layout="wide")
    
    st.title("📄 Document Similarity & Clustering Dashboard")
    st.markdown("Upload multiple documents to analyze topics, compute similarities, and discover clusters.")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Analysis Parameters")
    
    # Topic Modeling Parameters
    st.sidebar.subheader("📊 Topic Modeling (LDA)")
    num_topics = st.sidebar.slider("Number of Topics", min_value=2, max_value=15, value=5, step=1)
    num_words_per_topic = st.sidebar.slider("Words per Topic", min_value=5, max_value=20, value=10, step=1)
    
    with st.sidebar.expander("Advanced LDA Settings"):
        lda_passes = st.slider("Training Passes", min_value=5, max_value=50, value=10, step=5)
        lda_iterations = st.slider("Iterations per Pass", min_value=50, max_value=500, value=50, step=50)
        lda_alpha = st.selectbox("Alpha (Document-Topic Density)", ['auto', 'symmetric', 'asymmetric'], index=0)
        lda_eta = st.selectbox("Eta/Beta (Topic-Word Density)", ['auto', 'symmetric'], index=0)
    
    # Clustering Parameters
    st.sidebar.subheader("🎯 Clustering")
    clustering_algorithm = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Hierarchical"])
    
    if clustering_algorithm in ['KMeans', 'Hierarchical']:
        num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
    else:
        num_clusters = None
    
    if clustering_algorithm == 'DBSCAN':
        with st.sidebar.expander("DBSCAN Parameters"):
            dbscan_eps = st.slider("Epsilon (neighborhood size)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            dbscan_min_samples = st.slider("Min Samples", min_value=2, max_value=10, value=2, step=1)
    else:
        dbscan_eps = 0.5
        dbscan_min_samples = 2
    
    reduction_method = st.sidebar.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])
    
    # Session Management
    st.sidebar.divider()
    st.sidebar.subheader("💾 Session Management")
    
    # Load previous sessions
    view_mode = st.sidebar.radio("Mode", ["New Analysis", "Load Previous Session"])
    
    if view_mode == "Load Previous Session":
        sessions = load_analysis_sessions()
        if sessions:
            session_options = {f"{s.session_name} ({s.created_at.strftime('%Y-%m-%d %H:%M')})": s.id for s in sessions}
            selected_session_name = st.sidebar.selectbox("Select Session", list(session_options.keys()))
            
            if st.sidebar.button("📂 Load Session"):
                session_id = session_options[selected_session_name]
                loaded_session = load_session_by_id(session_id)
                
                if loaded_session:
                    # Store loaded data in the same session state structure as new analysis
                    loaded_params = json.loads(loaded_session.parameters)
                    
                    st.session_state['analysis_complete'] = True
                    st.session_state['current_documents'] = pickle.loads(loaded_session.documents_data)
                    st.session_state['current_doc_names'] = json.loads(loaded_session.document_names)
                    st.session_state['current_topic_distributions'] = pickle.loads(loaded_session.topic_distributions)
                    st.session_state['current_topics'] = pickle.loads(loaded_session.topics_data)
                    st.session_state['current_cluster_labels'] = pickle.loads(loaded_session.cluster_labels)
                    st.session_state['current_similarity_matrix'] = pickle.loads(loaded_session.similarity_matrix)
                    st.session_state['current_parameters'] = loaded_params
                    st.session_state['session_name_loaded'] = loaded_session.session_name
                    
                    # Need to compute reduced data since it wasn't saved
                    reduced_data = reduce_dimensions(
                        st.session_state['current_topic_distributions'],
                        method=loaded_params.get('reduction_method', 'PCA')
                    )
                    st.session_state['current_reduced_data'] = reduced_data
                    
                    st.success(f"✅ Loaded session: {loaded_session.session_name}")
                    st.rerun()
        else:
            st.sidebar.info("No saved sessions found")
    
    # Render analysis results if available (either from new analysis or loaded session)
    if st.session_state.get('analysis_complete', False):
        # Get data from session state
        documents = st.session_state['current_documents']
        doc_names = st.session_state['current_doc_names']
        topic_distributions = st.session_state['current_topic_distributions']
        topics = st.session_state['current_topics']
        cluster_labels = st.session_state['current_cluster_labels']
        similarity_matrix = st.session_state['current_similarity_matrix']
        reduced_data = st.session_state['current_reduced_data']
        params = st.session_state['current_parameters']
        
        num_topics = params['num_topics']
        clustering_algorithm = params['clustering_algorithm']
        reduction_method = params.get('reduction_method', 'PCA')
        
        # Show header
        if st.session_state.get('session_name_loaded'):
            st.header(f"📂 Loaded Session: {st.session_state['session_name_loaded']}")
        else:
            st.header("📊 Analysis Results")
        
        # Document info
        st.subheader("📋 Documents")
        doc_info = pd.DataFrame({
            'Document': doc_names,
            'Length (chars)': [len(doc) for doc in documents],
            'Word Count (approx)': [len(doc.split()) for doc in documents]
        })
        st.dataframe(doc_info, use_container_width=True)
        
        # Display topics summary
        st.subheader("📊 Discovered Topics")
        topic_cols = st.columns(min(3, num_topics))
        for idx in range(num_topics):
            col_idx = idx % len(topic_cols)
            with topic_cols[col_idx]:
                st.caption(f"**Topic {idx}**")
                words_str = ", ".join([word for word, _ in topics[idx][:3]])
                st.write(words_str)
    
    # Check if we're in load mode and no session loaded yet
    elif view_mode == "Load Previous Session":
        # Show session history
        st.header("📚 Session History")
        sessions = load_analysis_sessions()
        
        if sessions:
            # Create a table of sessions
            session_data = []
            for s in sessions:
                session_data.append({
                    'ID': s.id,
                    'Name': s.session_name,
                    'Date': s.created_at.strftime('%Y-%m-%d %H:%M'),
                    'Documents': s.num_documents,
                    'Topics': s.num_topics,
                    'Algorithm': s.clustering_algorithm
                })
            
            session_df = pd.DataFrame(session_data)
            st.dataframe(session_df, use_container_width=True, hide_index=True)
            
            st.info("💡 Select a session from the sidebar and click 'Load Session' to view details.")
        else:
            st.info("No saved sessions found. Complete an analysis and save it to build your history.")
    else:
        # File upload for new analysis
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload documents (.txt, .docx, .pdf, .rtf, .html, .md)",
            type=['txt', 'docx', 'doc', 'pdf', 'rtf', 'html', 'htm', 'md'],
            accept_multiple_files=True
        )
    
    # Handle new analysis
    if view_mode == "New Analysis" and uploaded_files and len(uploaded_files) >= 2:
        st.success(f"✅ {len(uploaded_files)} documents uploaded successfully!")
        
        # Parse documents
        with st.spinner("📖 Parsing documents..."):
            documents = []
            doc_names = []
            
            for file in uploaded_files:
                text = parse_document(file)
                if text:
                    documents.append(text)
                    doc_names.append(file.name)
            
            if len(documents) < 2:
                st.error("❌ Please upload at least 2 valid documents.")
                return
        
        # Display document info
        st.subheader("📋 Uploaded Documents")
        doc_info = pd.DataFrame({
            'Document': doc_names,
            'Length (chars)': [len(doc) for doc in documents],
            'Word Count (approx)': [len(doc.split()) for doc in documents]
        })
        st.dataframe(doc_info, use_container_width=True)
        
        # Perform analysis
        if st.button("🔍 Analyze Documents", type="primary"):
            
            # Validate num_clusters for algorithms that need it
            if clustering_algorithm in ['KMeans', 'Hierarchical'] and num_clusters > len(documents):
                st.error(f"❌ Number of clusters ({num_clusters}) cannot exceed the number of documents ({len(documents)}). Please adjust the slider in the sidebar.")
                return
            
            # LDA Topic Modeling
            try:
                with st.spinner("🧠 Performing LDA topic modeling..."):
                    lda_model, topic_distributions, topics, dictionary, corpus = perform_lda(
                        documents, 
                        num_topics=num_topics, 
                        num_words=num_words_per_topic,
                        passes=lda_passes,
                        iterations=lda_iterations,
                        alpha=lda_alpha,
                        eta=lda_eta
                    )
                
                st.success("✅ Topic modeling complete!")
            except ValueError as e:
                st.error(f"❌ Topic modeling failed: {str(e)}")
                st.info("💡 Try uploading documents with more content or adjust the number of topics.")
                return
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during topic modeling: {str(e)}")
                return
            
            # Display topics
            st.header("📊 Discovered Topics")
            cols = st.columns(2)
            for idx, topic_words in enumerate(topics):
                col_idx = idx % 2
                with cols[col_idx]:
                    st.subheader(f"Topic {idx}")
                    words_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words[:5]])
                    st.write(words_str)
            
            # Compute similarity
            with st.spinner("📐 Computing document similarities..."):
                similarity_matrix = compute_similarity_matrix(topic_distributions)
            
            # Clustering
            with st.spinner(f"🎯 Clustering documents using {clustering_algorithm}..."):
                cluster_labels, clusterer = perform_clustering(
                    topic_distributions, 
                    algorithm=clustering_algorithm,
                    num_clusters=num_clusters if num_clusters else 3,
                    eps=dbscan_eps,
                    min_samples=dbscan_min_samples
                )
            
            # Dimensionality reduction
            with st.spinner(f"🔬 Reducing dimensions using {reduction_method}..."):
                reduced_data = reduce_dimensions(topic_distributions, method=reduction_method)
            
            # Store results in session state for potential saving/reuse
            st.session_state['analysis_complete'] = True
            st.session_state['current_documents'] = documents
            st.session_state['current_doc_names'] = doc_names
            st.session_state['current_topic_distributions'] = topic_distributions
            st.session_state['current_topics'] = topics
            st.session_state['current_cluster_labels'] = cluster_labels
            st.session_state['current_similarity_matrix'] = similarity_matrix
            st.session_state['current_reduced_data'] = reduced_data
            st.session_state['current_parameters'] = {
                'num_topics': num_topics,
                'num_words_per_topic': num_words_per_topic,
                'lda_passes': lda_passes,
                'lda_iterations': lda_iterations,
                'lda_alpha': lda_alpha,
                'lda_eta': lda_eta,
                'clustering_algorithm': clustering_algorithm,
                'num_clusters': num_clusters,
                'dbscan_eps': dbscan_eps,
                'dbscan_min_samples': dbscan_min_samples,
                'reduction_method': reduction_method
            }
            
            # Visualizations
            st.header("📈 Visualizations")
            
            # Similarity Heatmap
            st.subheader("🔥 Document Similarity Heatmap")
            heatmap_fig = plot_similarity_heatmap(similarity_matrix, doc_names)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Cluster Plot
            st.subheader("🎨 Document Clusters")
            cluster_fig = plot_clusters(reduced_data, cluster_labels, doc_names, method=reduction_method)
            st.plotly_chart(cluster_fig, use_container_width=True)
            
            # Interactive Topic & Cluster Exploration
            st.header("🔍 Interactive Exploration")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📑 By Cluster", "📊 By Topic", "☁️ Word Clouds"])
            
            with tab1:
                st.subheader("Documents by Cluster")
                unique_clusters = sorted(set(cluster_labels))
                
                for cluster_id in unique_clusters:
                    cluster_docs_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_doc_names = [doc_names[i] for i in cluster_docs_indices]
                    
                    if cluster_id == -1:
                        cluster_title = "Noise/Outliers (DBSCAN)"
                    else:
                        cluster_title = f"Cluster {cluster_id}"
                    
                    with st.expander(f"{cluster_title} ({len(cluster_doc_names)} documents)", expanded=False):
                        for doc_idx in cluster_docs_indices:
                            doc_name = doc_names[doc_idx]
                            doc_text = documents[doc_idx]
                            
                            # Show document details
                            st.markdown(f"**📄 {doc_name}**")
                            
                            # Show topic distribution for this document
                            topic_dist = topic_distributions[doc_idx]
                            dominant_topics = sorted(enumerate(topic_dist), key=lambda x: x[1], reverse=True)[:3]
                            
                            topics_str = ", ".join([f"Topic {idx} ({prob:.2%})" for idx, prob in dominant_topics])
                            st.caption(f"Dominant Topics: {topics_str}")
                            
                            # Show document preview
                            preview_text = doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
                            st.text_area(f"Preview", preview_text, height=100, key=f"preview_{cluster_id}_{doc_idx}", disabled=True)
                            st.divider()
            
            with tab2:
                st.subheader("Documents by Dominant Topic")
                
                # Determine dominant topic for each document
                dominant_topic_per_doc = [np.argmax(dist) for dist in topic_distributions]
                
                for topic_idx in range(num_topics):
                    docs_in_topic = [i for i, dom_topic in enumerate(dominant_topic_per_doc) if dom_topic == topic_idx]
                    
                    if docs_in_topic:
                        topic_words_str = ", ".join([word for word, _ in topics[topic_idx][:5]])
                        
                        with st.expander(f"Topic {topic_idx}: {topic_words_str} ({len(docs_in_topic)} documents)", expanded=False):
                            for doc_idx in docs_in_topic:
                                doc_name = doc_names[doc_idx]
                                topic_strength = topic_distributions[doc_idx][topic_idx]
                                cluster_id = cluster_labels[doc_idx]
                                
                                st.markdown(f"**📄 {doc_name}** (Topic strength: {topic_strength:.2%}, Cluster: {cluster_id})")
                                
                                # Show document preview
                                preview_text = documents[doc_idx][:200] + "..." if len(documents[doc_idx]) > 200 else documents[doc_idx]
                                st.caption(preview_text)
                                st.divider()
            
            with tab3:
                st.subheader("Topic Word Clouds")
                num_cols = 2
                rows = (num_topics + num_cols - 1) // num_cols
                
                for row in range(rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        topic_idx = row * num_cols + col_idx
                        if topic_idx < num_topics:
                            with cols[col_idx]:
                                wc_fig = create_wordcloud(topics[topic_idx], topic_idx)
                                st.pyplot(wc_fig)
                                plt.close(wc_fig)
            
            # Downloads
            st.header("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            # Similarity Matrix CSV
            with col1:
                st.subheader("Similarity Scores")
                similarity_df = pd.DataFrame(similarity_matrix, columns=doc_names, index=doc_names)
                csv = similarity_df.to_csv()
                st.download_button(
                    label="📥 Download Similarity CSV",
                    data=csv,
                    file_name="similarity_matrix.csv",
                    mime="text/csv"
                )
            
            # Cluster Labels CSV
            with col2:
                st.subheader("Cluster Assignments")
                cluster_df = pd.DataFrame({
                    'Document': doc_names,
                    'Cluster': cluster_labels
                })
                csv = cluster_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Clusters CSV",
                    data=csv,
                    file_name="cluster_assignments.csv",
                    mime="text/csv"
                )
            
            # Topic Distributions CSV
            with col3:
                st.subheader("Topic Distributions")
                topic_dist_df = pd.DataFrame(
                    topic_distributions,
                    columns=[f'Topic_{i}' for i in range(num_topics)],
                    index=doc_names
                )
                csv = topic_dist_df.to_csv()
                st.download_button(
                    label="📥 Download Topics CSV",
                    data=csv,
                    file_name="topic_distributions.csv",
                    mime="text/csv"
                )
            
            # Word Cloud Images
            st.subheader("☁️ Word Cloud Images")
            wc_cols = st.columns(min(3, num_topics))
            for idx in range(num_topics):
                col_idx = idx % 3
                with wc_cols[col_idx]:
                    wc_fig = create_wordcloud(topics[idx], idx)
                    img_bytes = fig_to_bytes(wc_fig)
                    st.download_button(
                        label=f"📥 Topic {idx} Word Cloud",
                        data=img_bytes,
                        file_name=f"topic_{idx}_wordcloud.png",
                        mime="image/png"
                    )
                    plt.close(wc_fig)
            
            # Save Session Section
            st.header("💾 Save This Analysis")
            col_save1, col_save2 = st.columns([3, 1])
            
            with col_save1:
                session_name_input = st.text_input("Session Name", value=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            with col_save2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("💾 Save Session", type="secondary"):
                    parameters = {
                        'num_topics': num_topics,
                        'num_words_per_topic': num_words_per_topic,
                        'lda_passes': lda_passes,
                        'lda_iterations': lda_iterations,
                        'lda_alpha': lda_alpha,
                        'lda_eta': lda_eta,
                        'clustering_algorithm': clustering_algorithm,
                        'num_clusters': num_clusters,
                        'dbscan_eps': dbscan_eps,
                        'dbscan_min_samples': dbscan_min_samples,
                        'reduction_method': reduction_method
                    }
                    
                    if save_analysis_session(
                        session_name_input,
                        documents,
                        doc_names,
                        topic_distributions,
                        topics,
                        cluster_labels,
                        similarity_matrix,
                        parameters
                    ):
                        st.success(f"✅ Session '{session_name_input}' saved successfully!")
                    else:
                        st.error("❌ Failed to save session. Please check database connection.")
            
    elif view_mode == "New Analysis" and uploaded_files and len(uploaded_files) < 2:
        st.warning("⚠️ Please upload at least 2 documents to perform analysis.")
    else:
        st.info("👆 Upload at least 2 documents to get started.")
        
        # Example usage
        with st.expander("ℹ️ How to use this dashboard"):
            st.markdown("""
            ### Steps:
            1. **Upload Documents**: Upload at least 2 documents in .txt, .docx, or .pdf format
            2. **Adjust Parameters**: Use the sidebar to configure:
               - Number of topics for LDA analysis
               - Number of clusters for KMeans
               - Dimensionality reduction method (PCA or t-SNE)
               - Words displayed per topic
            3. **Analyze**: Click the "Analyze Documents" button
            4. **Review Results**: Explore the discovered topics, similarity heatmap, and cluster visualization
            5. **Download**: Export similarity scores, cluster assignments, and word cloud images
            
            ### What this dashboard does:
            - **Topic Modeling**: Uses LDA (Latent Dirichlet Allocation) to discover hidden topics in your documents
            - **Similarity Analysis**: Computes how similar documents are based on their topic distributions
            - **Clustering**: Groups similar documents together using KMeans algorithm
            - **Visualization**: Provides interactive charts and word clouds for easy interpretation
            """)

if __name__ == "__main__":
    main()
