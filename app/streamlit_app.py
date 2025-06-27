import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="Intelligent Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .question-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .answer-card {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Please ensure the server is running on http://localhost:8000")
        return {"error": "Connection failed"}
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return {"error": str(e)}

def upload_documents_tab():
    """Document upload interface"""
    st.header("📁 Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to upload",
        type=['txt', 'pdf', 'docx', 'md'],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, DOCX, MD"
    )
    
    if uploaded_files:
        if st.button("🚀 Upload and Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Prepare files for upload
                files = []
                for uploaded_file in uploaded_files:
                    files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
                
                # Make API request
                result = make_api_request("/upload-documents", method="POST", files=dict(files))
                
                if "error" not in result:
                    st.session_state.uploaded_documents = result.get("documents", [])
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-message">
                        <h4>✅ Upload Successful!</h4>
                        <p><strong>Total Files:</strong> {result['summary']['total_files']}</p>
                        <p><strong>Successfully Processed:</strong> {result['summary']['successful']}</p>
                        <p><strong>Failed:</strong> {result['summary']['failed']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display processed documents
                    if result.get("documents"):
                        st.subheader("📋 Processing Results")
                        for doc in result["documents"]:
                            if doc["status"] == "processed":
                                st.success(f"✅ {doc['filename']} - {doc['chunks']} chunks, {doc['processing_time']}s")
                            else:
                                st.error(f"❌ {doc['filename']} - {doc.get('error', 'Unknown error')}")
    
    # Display current documents
    st.subheader("📚 Current Documents")
    documents_result = make_api_request("/list-documents")
    
    if "error" not in documents_result and documents_result.get("documents"):
        df = pd.DataFrame(documents_result["documents"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No documents uploaded yet.")

def question_generation_tab():
    """Question generation interface"""
    st.header("❓ Question Generation")
    
    # Get available documents
    documents_result = make_api_request("/list-documents")
    
    if "error" not in documents_result and documents_result.get("documents"):
        documents = documents_result["documents"]
        
        # Document selection
        doc_options = {f"{doc['filename']} ({doc['chunk_count']} chunks)": doc['doc_id'] 
                      for doc in documents}
        
        selected_doc = st.selectbox(
            "Select a document for question generation:",
            options=list(doc_options.keys()),
            help="Choose a document to generate questions from"
        )
        
        if selected_doc:
            doc_id = doc_options[selected_doc]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🎯 Generate Questions", type="primary"):
                    with st.spinner("Generating questions..."):
                        result = make_api_request(f"/generate-questions/{doc_id}")
                        
                        if "error" not in result:
                            st.session_state[f"questions_{doc_id}"] = result
                            st.success(f"✅ Generated {result['questions_generated']} questions!")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Display generated questions
            if f"questions_{doc_id}" in st.session_state:
                questions_data = st.session_state[f"questions_{doc_id}"]
                
                st.subheader("📝 Generated Questions")
                
                for i, question in enumerate(questions_data["questions"], 1):
                    st.markdown(f"""
                    <div class="question-card">
                        <strong>Question {i}:</strong> {question}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Evaluation button
                if st.button("📊 Run Evaluation", type="secondary"):
                    with st.spinner("Running evaluation..."):
                        eval_result = make_api_request(f"/evaluate/{doc_id}")
                        
                        if "error" not in eval_result:
                            st.session_state.evaluation_results[doc_id] = eval_result["results"]
                            st.success("✅ Evaluation completed!")
                            st.rerun()
                        else:
                            st.error(f"❌ Evaluation failed: {eval_result.get('error', 'Unknown error')}")
    else:
        st.info("📤 Please upload documents first to generate questions.")

def evaluation_results_tab():
    """Evaluation results display"""
    st.header("📊 Evaluation Results")
    
    if st.session_state.evaluation_results:
        # Document selection for results
        doc_ids = list(st.session_state.evaluation_results.keys())
        selected_doc_id = st.selectbox("Select document results:", doc_ids)
        
        if selected_doc_id:
            results = st.session_state.evaluation_results[selected_doc_id]
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['average_f1']:.3f}</h3>
                    <p>Average F1 Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['average_semantic']:.3f}</h3>
                    <p>Semantic Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['accuracy_rate']:.1%}</h3>
                    <p>Accuracy Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['average_response_time']:.2f}s</h3>
                    <p>Avg Response Time</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed results
            st.subheader("📋 Detailed Results")
            
            if results.get("predictions"):
                # Create DataFrame for detailed results
                predictions_df = pd.DataFrame(results["predictions"])
                
                # Display results table
                st.dataframe(
                    predictions_df[['question', 'f1_score', 'semantic_score', 'contains_answer', 'response_time']],
                    use_container_width=True
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # F1 Score distribution
                    fig_f1 = px.histogram(
                        predictions_df, 
                        x='f1_score', 
                        title='F1 Score Distribution',
                        nbins=10
                    )
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                with col2:
                    # Question types
                    if results.get("question_types"):
                        fig_types = px.pie(
                            values=list(results["question_types"].values()),
                            names=list(results["question_types"].keys()),
                            title='Question Types Distribution'
                        )
                        st.plotly_chart(fig_types, use_container_width=True)
                
                # Individual question analysis
                st.subheader("🔍 Individual Question Analysis")
                
                for i, pred in enumerate(results["predictions"]):
                    with st.expander(f"Question {i+1} (F1: {pred['f1_score']:.3f})"):
                        st.markdown(f"""
                        <div class="question-card">
                            <strong>Question:</strong> {pred['question']}
                        </div>
                        <div class="answer-card">
                            <strong>System Answer:</strong> {pred['predicted']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("F1 Score", f"{pred['f1_score']:.3f}")
                        with col2:
                            st.metric("Semantic Score", f"{pred['semantic_score']:.3f}")
                        with col3:
                            st.metric("Response Time", f"{pred['response_time']:.2f}s")
    else:
        st.info("📊 No evaluation results available. Please generate questions and run evaluation first.")

def chat_interface_tab():
    """Interactive chat interface"""
    st.header("💬 Interactive Q&A")
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What would you like to know?",
        key="chat_input"
    )
    
    if user_question and st.button("Send", type="primary"):
        with st.spinner("Generating answer..."):
            # Make API request
            request_data = {
                "question": user_question,
                "session_id": st.session_state.session_id
            }
            
            result = make_api_request("/ask-question", method="POST", data=request_data)
            
            if "error" not in result:
                # Update session ID
                st.session_state.session_id = result.get("session_id")
                
                # Add to conversation history
                interaction = {
                    "question": user_question,
                    "answer": result["answer"],
                    "confidence": result.get("confidence", 0),
                    "sources": result.get("sources", []),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.conversation_history.append(interaction)
                
                # Clear input
                st.rerun()
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("💭 Conversation History")
        
        for i, interaction in enumerate(reversed(st.session_state.conversation_history)):
            with st.container():
                st.markdown(f"""
                <div class="question-card">
                    <strong>Q:</strong> {interaction['question']}
                </div>
                <div class="answer-card">
                    <strong>A:</strong> {interaction['answer']}
                    <br><small>Confidence: {interaction['confidence']:.2f} | {interaction['timestamp'][:19]}</small>
                </div>
                """, unsafe_allow_html=True)
                
                if interaction.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in interaction["sources"]:
                            st.text(source)

def system_stats_tab():
    """System statistics and monitoring"""
    st.header("📈 System Statistics")
    
    # Get system stats
    stats_result = make_api_request("/system-stats")
    
    if "error" not in stats_result:
        stats = stats_result
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Processed", stats["system"]["documents_processed"])
        with col2:
            st.metric("Chunks Indexed", stats["system"]["chunks_indexed"])
        with col3:
            st.metric("Active Sessions", stats["sessions"]["active_sessions"])
        
        # Document processing stats
        if stats.get("documents"):
            st.subheader("📊 Document Processing Statistics")
            
            # Create processing stats visualization
            doc_stats = []
            for doc_id, doc_info in stats["documents"].items():
                doc_stats.append({
                    "Document": doc_info.get("filename", doc_id),
                    "Chunks": len(doc_info.get("chunks", [])),
                    "Text Length": doc_info.get("text_length", 0)
                })
            
            if doc_stats:
                df_stats = pd.DataFrame(doc_stats)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_chunks = px.bar(df_stats, x="Document", y="Chunks", title="Chunks per Document")
                    st.plotly_chart(fig_chunks, use_container_width=True)
                
                with col2:
                    fig_length = px.bar(df_stats, x="Document", y="Text Length", title="Text Length per Document")
                    st.plotly_chart(fig_length, use_container_width=True)
        
        # Session statistics
        st.subheader("👥 Session Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Interactions", stats["sessions"]["total_interactions"])
        with col2:
            st.metric("Cache Size", stats["system"]["cache_size"])
    
    # Health check
    st.subheader("🏥 System Health")
    health_result = make_api_request("/health")
    
    if "error" not in health_result:
        st.success("✅ System is healthy")
        st.json(health_result)
    else:
        st.error("❌ System health check failed")

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🤖 Intelligent Q&A System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    tabs = [
        "📁 Upload Documents",
        "❓ Generate Questions", 
        "📊 Evaluation Results",
        "💬 Interactive Q&A",
        "📈 System Stats"
    ]
    
    selected_tab = st.sidebar.radio("Select a tab:", tabs)
    
    # Tab routing
    if selected_tab == "📁 Upload Documents":
        upload_documents_tab()
    elif selected_tab == "❓ Generate Questions":
        question_generation_tab()
    elif selected_tab == "📊 Evaluation Results":
        evaluation_results_tab()
    elif selected_tab == "💬 Interactive Q&A":
        chat_interface_tab()
    elif selected_tab == "📈 System Stats":
        system_stats_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Intelligent Q&A System v1.0**")
    st.sidebar.markdown("Built with Streamlit & FastAPI")

if __name__ == "__main__":
    main()
