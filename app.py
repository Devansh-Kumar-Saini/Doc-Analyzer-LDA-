from fastapi import FastAPI
health_app = FastAPI()

@health_app.get("/healthz")
def health_check():
    return {"status": "ok"}

# Existing Streamlit code...
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

def plot_clusters(reduced_data, cluster_labels, doc_names, method='PCA', theme=None, color_palette=None):
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
    
    if theme and color_palette:
        return apply_theme_settings(fig, theme, color_palette)
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

# New visualization functions
def plot_topic_distributions(topic_distributions, doc_names, topics):
    """Create interactive bar chart of topic distributions per document"""
    df = pd.DataFrame(topic_distributions, columns=[f"Topic {i}" for i in range(len(topics))])
    df['Document'] = doc_names
    df = df.melt(id_vars='Document', var_name='Topic', value_name='Proportion')
    
    fig = px.bar(
        df, 
        x='Document', 
        y='Proportion', 
        color='Topic',
        title='Topic Distributions per Document',
        labels={'Proportion': 'Topic Proportion'},
        height=600
    )
    fig.update_layout(barmode='stack', xaxis_tickangle=-45)
    return fig

def plot_similarity_network(similarity_matrix, doc_names, threshold=0.5):
    """Create interactive network graph of document similarities"""
    edges = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                edges.append((i, j, similarity_matrix[i][j]))
    
    if not edges:
        return None
    
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = np.random.rand(2)
        x1, y1 = np.random.rand(2)
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for i in range(len(doc_names)):
        node_x.append(np.random.rand())
        node_y.append(np.random.rand())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=doc_names,
        textposition='bottom center',
        hovertext=[f"Connections: {sum(similarity_matrix[i] > threshold)-1}" for i in range(len(doc_names))],
        marker=dict(
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Document Similarity Network',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

# Download functions
def get_csv_download_link(df, filename):
    """Generate download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def fig_to_bytes(fig):
    """Convert matplotlib or plotly figure to bytes"""
    if hasattr(fig, 'savefig'):  # Matplotlib figure
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf
    else:  # Plotly figure
        try:
            return fig.to_image(format='png', width=800, height=600, scale=2)
        except ValueError:
            # Fallback to SVG if Kaleido not available
            return fig.to_image(format='svg')

# Main Streamlit App
def main():
    st.set_page_config(page_title="Document Similarity & Clustering Dashboard", layout="wide")
    
    # Initialize session state variables
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'preserve_analysis' not in st.session_state:
        st.session_state.preserve_analysis = False

    # Initialize regular variables
    uploaded_files = []
    
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
    
    # Visualization Settings
    st.sidebar.subheader("🎨 Visualization Settings")
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)
    color_palette = st.sidebar.selectbox("Color Palette", 
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"], 
        index=0)

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
        
        # Initialize empty list if no files uploaded
        if uploaded_files is None:
            uploaded_files = []
    
    # Handle new analysis
    if view_mode == "New Analysis":
        if len(uploaded_files) >= 2:
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
            if st.button("🔍 Analyze Documents", type="primary") or st.session_state.preserve_analysis:
                st.session_state.preserve_analysis = True
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
                
                # Store results in session state first
                st.session_state['analysis_results'] = {
                    'documents': documents,
                    'doc_names': doc_names,
                    'topic_distributions': topic_distributions,
                    'topics': topics,
                    'cluster_labels': cluster_labels,
                    'similarity_matrix': similarity_matrix,
                    'reduced_data': reduced_data,
                    'params': {
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
                }
                
                # Downloads section moved BEFORE visualizations
                st.header("💾 Download Options")
                with st.expander("Click to show download options", expanded=False):
                    dl_tab1, dl_tab2, dl_tab3 = st.tabs(["📷 Images", "📄 HTML", "📚 Full Report"])
                    
                    with dl_tab1:
                        st.subheader("Download Visualizations as Images")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            heatmap_fig = plot_similarity_heatmap(similarity_matrix, doc_names)
                            st.download_button(
                                label="📥 Heatmap",
                                data=fig_to_bytes(heatmap_fig),
                                file_name="similarity_heatmap.png",
                                mime="image/png",
                                key="dl_heatmap"
                            )
                        
                        with col2:
                            cluster_fig = plot_clusters(reduced_data, cluster_labels, doc_names, method=reduction_method, theme=theme, color_palette=color_palette)
                            st.download_button(
                                label="📥 Cluster Plot",
                                data=fig_to_bytes(cluster_fig),
                                file_name="document_clusters.png",
                                mime="image/png",
                                key="dl_cluster"
                            )
                        
                        with col3:
                            network_fig = plot_similarity_network(similarity_matrix, doc_names, threshold=0.5)
                            if network_fig:
                                st.download_button(
                                    label="📥 Network Graph",
                                    data=fig_to_bytes(network_fig),
                                    file_name="similarity_network.png",
                                    mime="image/png",
                                    key="dl_network"
                                )
                
                # Then show visualizations
                st.header("📈 Analysis Results")
                # Generate truly unique analysis ID
                import time
                unique_id = f"{int(time.time())}_{st.session_state.analysis_count}"
                
                # Topic Distribution Bar Chart
                st.subheader(f"📊 Topic Distributions (Analysis #{st.session_state.analysis_count})")
                topic_dist_fig = plot_topic_distributions(topic_distributions, doc_names, topics)
                st.plotly_chart(topic_dist_fig, use_container_width=True, key=f"topic_dist_{unique_id}")
                
                # Similarity Network Graph
                st.subheader(f"🕸️ Document Similarity Network (Analysis #{st.session_state.analysis_count})")
                network_threshold = st.slider(
                    "Similarity Threshold", 
                    min_value=0.1, 
                    max_value=0.9, 
                    value=st.session_state.get('last_threshold', 0.5), 
                    step=0.1,
                    help="Adjust to show more/less connections",
                    key=f"network_threshold_{unique_id}",
                    on_change=lambda: st.session_state.update({'last_threshold': st.session_state[f"network_threshold_{unique_id}"]})
                )
                network_fig = plot_similarity_network(similarity_matrix, doc_names, threshold=network_threshold)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True, key=f"network_{unique_id}")
                
                # Heatmap
                st.subheader(f"🔥 Document Similarity Heatmap (Analysis #{st.session_state.analysis_count})")
                heatmap_fig = plot_similarity_heatmap(similarity_matrix, doc_names)
                st.plotly_chart(heatmap_fig, use_container_width=True, key=f"heatmap_{unique_id}")
                
                # Cluster Plot
                st.subheader(f"🎨 Document Clusters (Analysis #{st.session_state.analysis_count})")
                cluster_fig = plot_clusters(reduced_data, cluster_labels, doc_names, method=reduction_method, theme=theme, color_palette=color_palette)
                st.plotly_chart(cluster_fig, use_container_width=True, key=f"clusters_{unique_id}")
                
                # Enhanced Visualizations Section
                st.header("📈 Enhanced Visualizations")
                
                # Topic Distribution Bar Chart
                st.subheader(f"📊 Topic Distributions (Analysis #{st.session_state.analysis_count})")
                topic_dist_fig = plot_topic_distributions(topic_distributions, doc_names, topics)
                st.plotly_chart(topic_dist_fig, use_container_width=True, key=f"topic_dist_chart_{st.session_state.analysis_count}")
                
                # Similarity Network Graph
                st.subheader(f"🕸️ Document Similarity Network (Analysis #{st.session_state.analysis_count})")
                network_threshold = st.slider(
                    "Similarity Threshold", 
                    min_value=0.1, 
                    max_value=0.9, 
                    value=st.session_state.get('last_threshold', 0.5), 
                    step=0.1,
                    help="Adjust to show more/less connections",
                    key=f"network_threshold_{st.session_state.analysis_count}"
                )
                network_fig = plot_similarity_network(similarity_matrix, doc_names, threshold=network_threshold)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True, key=f"network_chart_{st.session_state.analysis_count}")
                
                # Heatmap
                st.subheader(f"🔥 Document Similarity Heatmap (Analysis #{st.session_state.analysis_count})")
                heatmap_fig = plot_similarity_heatmap(similarity_matrix, doc_names)
                st.plotly_chart(heatmap_fig, use_container_width=True, key=f"heatmap_chart_{st.session_state.analysis_count}")
                
                # Cluster Plot
                st.subheader(f"🎨 Document Clusters (Analysis #{st.session_state.analysis_count})")
                cluster_fig = plot_clusters(reduced_data, cluster_labels, doc_names, method=reduction_method, theme=theme, color_palette=color_palette)
                st.plotly_chart(cluster_fig, use_container_width=True, key=f"cluster_chart_{st.session_state.analysis_count}")
                
                # Enhanced Downloads Section
                st.header("💾 Enhanced Download Options")
                
                # Create tabs for different download formats
                dl_tab1, dl_tab2, dl_tab3 = st.tabs(["📷 Images", "📄 HTML", "📚 Full Report"])
                
                with dl_tab1:
                    st.subheader("Download Visualizations as Images")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📥 Heatmap",
                            data=fig_to_bytes(heatmap_fig),
                            file_name="similarity_heatmap.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📥 Cluster Plot",
                            data=fig_to_bytes(cluster_fig),
                            file_name="document_clusters.png",
                            mime="image/png"
                        )
                    
                    with col3:
                        if network_fig:
                            st.download_button(
                                label="📥 Network Graph",
                                data=fig_to_bytes(network_fig),
                                file_name="similarity_network.png",
                                mime="image/png"
                            )
                
                with dl_tab2:
                    st.subheader("Download Interactive HTML")
                    st.warning("HTML export coming soon! This will allow you to save interactive versions of the visualizations.")
                
                with dl_tab3:
                    st.subheader("Generate Full Report")
                    st.warning("PDF report generation coming soon! This will compile all analysis results into a single PDF.")
                
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
            
    elif view_mode == "New Analysis" and uploaded_files and len(uploaded_files) == 1:
        st.error("❌ Please upload at least 2 documents for analysis.")
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

def apply_theme_settings(fig, theme, color_palette):
    """Apply theme and color settings to figures"""
    if theme == "Dark":
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    else:
        fig.update_layout(template="plotly_white")
    
    # Apply color palette based on plot type
    for trace in fig.data:
        if trace.type == 'scatter':
            trace.update(marker=dict(colorscale=color_palette.lower()))
        elif trace.type == 'heatmap':
            trace.update(colorscale=color_palette.lower())
    
    return fig

if __name__ == "__main__":
    main()
