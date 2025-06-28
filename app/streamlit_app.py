import streamlit as st
import asyncio
import logging
import time
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any

# Import our modules
from .main import QASystem
from .simple_evaluator import SimpleEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = "http://127.0.0.1:8000"

# Configure Streamlit page
st.set_page_config(
    page_title="Intelligent Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .session-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .session-active {
        background-color: #28a745;
    }
    .session-inactive {
        background-color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()
    
    if 'simple_evaluator' not in st.session_state:
        st.session_state.simple_evaluator = SimpleEvaluator(
            st.session_state.qa_system.rag_system,
            st.session_state.qa_system.document_processor
        )
    
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = {}
    
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    
    # Session management
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = f"session_{int(time.time())}"
    
    if 'all_sessions' not in st.session_state:
        st.session_state.all_sessions = {}
    
    # Initialize current session if not exists
    if st.session_state.current_session_id not in st.session_state.all_sessions:
        st.session_state.all_sessions[st.session_state.current_session_id] = {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'conversation_history': [],
            'name': f"Session {len(st.session_state.all_sessions) + 1}"
        }

def create_new_session():
    """Create a new conversation session"""
    new_session_id = f"session_{int(time.time())}"
    st.session_state.current_session_id = new_session_id
    st.session_state.all_sessions[new_session_id] = {
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'conversation_history': [],
        'name': f"Session {len(st.session_state.all_sessions) + 1}"
    }
    st.rerun()

def get_current_conversation():
    """Get current session's conversation history"""
    return st.session_state.all_sessions[st.session_state.current_session_id]['conversation_history']

def add_to_conversation(question: str, answer: str, confidence: float = 0.0):
    """Add Q&A to current session's conversation history"""
    conversation = get_current_conversation()
    conversation.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': answer,
        'confidence': confidence
    })

async def process_question(question: str) -> Dict[str, Any]:
    """Process a question using the QA system"""
    try:
        conversation_history = get_current_conversation()
        result = await st.session_state.qa_system.ask_question(question, conversation_history)
        
        # Add to conversation history
        add_to_conversation(question, result['answer'], result.get('confidence', 0.0))
        
        return result
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {
            'answer': f"I encountered an error while processing your question: {str(e)}",
            'confidence': 0.0,
            'sources': [],
            'error': str(e)
        }

def render_sidebar():
    """Render the sidebar with navigation and session management"""
    with st.sidebar:
        st.markdown("### 🤖 Navigation")
        
        # Session Management
        st.markdown("### 💬 Session Management")
        
        # Current session indicator
        current_session = st.session_state.all_sessions[st.session_state.current_session_id]
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span class="session-indicator session-active"></span>
            <strong>Current:</strong> {current_session['name']}<br>
            <small>Created: {current_session['created_at']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # New session button
        if st.button("🆕 New Session", use_container_width=True):
            create_new_session()
        
        # Session selector
        if len(st.session_state.all_sessions) > 1:
            st.markdown("**Switch Session:**")
            session_options = {}
            for session_id, session_data in st.session_state.all_sessions.items():
                is_current = session_id == st.session_state.current_session_id
                indicator = "🟢" if is_current else "⚪"
                session_options[f"{indicator} {session_data['name']}"] = session_id
            
            selected_session_name = st.selectbox(
                "Select session:",
                options=list(session_options.keys()),
                index=list(session_options.values()).index(st.session_state.current_session_id),
                label_visibility="collapsed"
            )
            
            selected_session_id = session_options[selected_session_name]
            if selected_session_id != st.session_state.current_session_id:
                st.session_state.current_session_id = selected_session_id
                st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Choose a page:",
            ["📄 Document Upload", "❓ Ask Questions", "📊 Evaluation", "📈 History"],
            index=0
        )
        
        # System status
        st.markdown("### 📊 System Status")
        
        # Document count
        doc_count = len(st.session_state.uploaded_documents)
        st.metric("📚 Documents", doc_count)
        
        # Current session stats
        conversation_count = len(get_current_conversation())
        st.metric("💬 Questions (Current)", conversation_count)
        
        # Total conversations across all sessions
        total_conversations = sum(len(session['conversation_history']) for session in st.session_state.all_sessions.values())
        st.metric("💬 Total Questions", total_conversations)
        
        return page

def render_document_upload():
    """Render document upload page"""
    st.markdown('<h1 class="main-header">📄 Document Upload</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload your documents to build the knowledge base for the Q&A system.
    Supported formats: PDF, TXT, DOCX
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_documents:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Process the document
                        result = asyncio.run(st.session_state.qa_system.upload_document(uploaded_file))
                        
                        if result.get('success'):
                            st.session_state.uploaded_documents[uploaded_file.name] = {
                                'doc_id': result['doc_id'],
                                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'chunks': result.get('chunks_created', 0),
                                'size': uploaded_file.size
                            }
                            
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                            st.info(f"📊 Created {result.get('chunks_created', 0)} chunks")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.markdown("### 📚 Uploaded Documents")
        
        for filename, doc_info in st.session_state.uploaded_documents.items():
            with st.expander(f"📄 {filename}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Chunks", doc_info['chunks'])
                
                with col2:
                    st.metric("📅 Uploaded", doc_info['upload_time'])
                
                with col3:
                    size_mb = doc_info['size'] / (1024 * 1024)
                    st.metric("💾 Size", f"{size_mb:.2f} MB")

def render_ask_questions():
    """Render question asking page"""
    st.markdown('<h1 class="main-header">❓ Ask Questions</h1>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload documents first before asking questions.")
        return
    
    # Current session info
    current_session = st.session_state.all_sessions[st.session_state.current_session_id]
    st.info(f"💬 Current Session: **{current_session['name']}** (Created: {current_session['created_at']})")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about the uploaded documents?",
        help="Ask any question about the content in your uploaded documents"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("🤔 Thinking..."):
            result = asyncio.run(process_question(question))
            
            # Display answer
            st.markdown("### 💡 Answer")
            
            if 'error' not in result:
                st.markdown(f"**Confidence:** {result.get('confidence', 0):.2f}")
                st.markdown(result['answer'])
                
                # Display sources if available
                if result.get('sources'):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"**Source {i}:** {source.get('source', 'Unknown')}")
                            st.markdown(f"*Similarity:* {source.get('similarity', 0):.3f}")
                            st.markdown(f"*Preview:* {source.get('preview', 'No preview available')}")
                            st.markdown("---")
            else:
                st.error(f"❌ {result['answer']}")
    
    # Display conversation history for current session
    conversation = get_current_conversation()
    if conversation:
        st.markdown("### 💬 Conversation History (Current Session)")
        
        for i, item in enumerate(reversed(conversation[-10:])):  # Show last 10
            with st.expander(f"Q{len(conversation)-i}: {item['question'][:50]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown(f"**Confidence:** {item['confidence']:.2f}")
                st.markdown(f"**Time:** {item['timestamp']}")

def render_evaluation():
    """Render evaluation page with Gemini-powered question generation"""
    st.markdown('<h1 class="main-header">📊 Evaluation</h1>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload documents first before running evaluation.")
        return
    
    st.markdown("""
    ### 🧠 AI-Powered Evaluation
    
    This evaluation uses **Gemini AI** to generate intelligent, context-aware questions from your documents.
    The system will:
    - 🤖 Generate diverse questions using Gemini's language understanding
    - 📝 Create factual, analytical, and conceptual questions
    - 🎯 Test the RAG system's ability to answer these AI-generated questions
    - 📊 Provide detailed performance metrics
    """)
    
    # Document selection
    doc_options = list(st.session_state.uploaded_documents.keys())
    selected_doc = st.selectbox("Select document for evaluation:", doc_options)
    
    if selected_doc:
        doc_info = st.session_state.uploaded_documents[selected_doc]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Chunks", doc_info['chunks'])
        with col2:
            st.metric("📅 Uploaded", doc_info['upload_time'])
        with col3:
            # Estimate questions based on chunks
            estimated_questions = min(8, max(5, doc_info['chunks'] // 3))
            st.metric("🤖 Est. Questions", estimated_questions)
        
        # Evaluation settings
        st.markdown("### ⚙️ Evaluation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            max_questions = st.slider(
                "Maximum questions to generate:",
                min_value=5,
                max_value=10,
                value=min(8, max(5, doc_info['chunks'] // 3)),
                help="Gemini will generate up to this many questions"
            )
        
        with col2:
            st.info("🤖 **Gemini AI Features:**\n- Context-aware questions\n- Multiple question types\n- Natural language generation\n- Intelligent fallbacks")
        
        # Run evaluation button
        if st.button("🚀 Run AI Evaluation", type="primary", use_container_width=True):
            with st.spinner("🤖 Gemini is generating intelligent questions..."):
                try:
                    # Run evaluation
                    doc_id = doc_info['doc_id']
                    
                    # Generate questions using Gemini
                    st.info("🧠 Step 1: Gemini AI is analyzing document content...")
                    questions_data = asyncio.run(
                        st.session_state.simple_evaluator.generate_questions_from_document(
                            doc_id, max_questions
                        )
                    )
                    
                    if not questions_data:
                        st.error("❌ Failed to generate questions from the document.")
                        return
                    
                    st.success(f"✅ Generated {len(questions_data)} AI-powered questions!")
                    
                    # Show generated questions
                    with st.expander("🤖 View Gemini-Generated Questions"):
                        for i, q_data in enumerate(questions_data, 1):
                            st.markdown(f"**Q{i} ({q_data.get('question_type', 'general')}):** {q_data['question']}")
                    
                    # Run evaluation
                    st.info("📊 Step 2: Evaluating RAG system performance...")
                    evaluation_results = asyncio.run(
                        st.session_state.simple_evaluator.evaluate_with_f1(questions_data)
                    )
                    
                    # Store results
                    st.session_state.evaluation_results[selected_doc] = evaluation_results
                    
                    st.success("🎉 Evaluation completed!")
                    
                    # Display results
                    render_evaluation_results(evaluation_results)
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    logger.error(f"Evaluation error: {str(e)}")
    
    # Display previous evaluation results
    if st.session_state.evaluation_results:
        st.markdown("### 📈 Previous Evaluation Results")
        
        for doc_name, results in st.session_state.evaluation_results.items():
            with st.expander(f"📊 Results for {doc_name}"):
                render_evaluation_results(results)

def render_evaluation_results(results: Dict[str, Any]):
    """Render evaluation results with enhanced metrics"""
    if 'error' in results:
        st.error(f"❌ Evaluation error: {results['error']}")
        return
    
    # Key metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Average F1 Score",
            f"{results['average_f1']:.3f}",
            help="F1 score measures answer quality (0-1, higher is better)"
        )
    
    with col2:
        st.metric(
            "🧠 Semantic Similarity",
            f"{results['average_semantic']:.3f}",
            help="Semantic similarity between expected and actual answers"
        )
    
    with col3:
        st.metric(
            "✅ Accuracy Rate",
            f"{results['accuracy_rate']:.1%}",
            help="Percentage of questions answered correctly"
        )
    
    with col4:
        st.metric(
            "⚡ Avg Response Time",
            f"{results['average_response_time']:.2f}s",
            help="Average time to generate answers"
        )
    
    # Question types breakdown
    if results.get('question_types'):
        st.markdown("### 🤖 AI Question Types Generated")
        
        question_types_df = pd.DataFrame(
            list(results['question_types'].items()),
            columns=['Question Type', 'Count']
        )
        
        fig = px.pie(
            question_types_df,
            values='Count',
            names='Question Type',
            title="Distribution of AI-Generated Question Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance distribution
    if results.get('f1_scores'):
        st.markdown("### 📈 Performance Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # F1 scores histogram
            fig_f1 = px.histogram(
                x=results['f1_scores'],
                nbins=10,
                title="F1 Score Distribution",
                labels={'x': 'F1 Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_f1, use_container_width=True)
        
        with col2:
            # Response times
            fig_time = px.histogram(
                x=results['response_times'],
                nbins=10,
                title="Response Time Distribution",
                labels={'x': 'Response Time (seconds)', 'y': 'Count'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
    
    # Detailed predictions
    if results.get('predictions'):
        st.markdown("### 🔍 Detailed Question Analysis")
        
        predictions_df = pd.DataFrame(results['predictions'])
        
        # Add performance categories
        def categorize_performance(f1_score):
            if f1_score >= 0.8:
                return "🟢 Excellent"
            elif f1_score >= 0.6:
                return "🟡 Good"
            elif f1_score >= 0.4:
                return "🟠 Fair"
            else:
                return "🔴 Poor"
        
        predictions_df['Performance'] = predictions_df['f1_score'].apply(categorize_performance)
        
        # Display table
        st.dataframe(
            predictions_df[['question', 'question_type', 'f1_score', 'semantic_score', 'contains_answer', 'Performance']],
            use_container_width=True
        )
        
        # Show individual predictions
        with st.expander("📝 View Individual Q&A Pairs"):
            for i, pred in enumerate(results['predictions'], 1):
                st.markdown(f"**Question {i} ({pred.get('question_type', 'general')}):**")
                st.markdown(f"*Q:* {pred['question']}")
                st.markdown(f"*Expected:* {pred['expected'][:200]}...")
                st.markdown(f"*Predicted:* {pred['predicted'][:200]}...")
                st.markdown(f"*F1 Score:* {pred['f1_score']:.3f}")
                st.markdown("---")

def render_history():
    """Render conversation history across all sessions"""
    st.markdown('<h1 class="main-header">📈 Conversation History</h1>', unsafe_allow_html=True)
    
    if not st.session_state.all_sessions:
        st.info("No conversation history available yet.")
        return
    
    # Session overview
    st.markdown("### 💬 Session Overview")
    
    session_data = []
    for session_id, session_info in st.session_state.all_sessions.items():
        is_current = session_id == st.session_state.current_session_id
        session_data.append({
            'Session': session_info['name'],
            'Created': session_info['created_at'],
            'Questions': len(session_info['conversation_history']),
            'Status': '🟢 Current' if is_current else '⚪ Archived'
        })
    
    if session_data:
        sessions_df = pd.DataFrame(session_data)
        st.dataframe(sessions_df, use_container_width=True)
    
    # Session selector for detailed view
    st.markdown("### 🔍 Detailed Session History")
    
    session_options = {}
    for session_id, session_data in st.session_state.all_sessions.items():
        is_current = session_id == st.session_state.current_session_id
        indicator = "🟢" if is_current else "⚪"
        session_options[f"{indicator} {session_data['name']} ({len(session_data['conversation_history'])} questions)"] = session_id
    
    if session_options:
        selected_session_name = st.selectbox(
            "Select session to view:",
            options=list(session_options.keys())
        )
        
        selected_session_id = session_options[selected_session_name]
        selected_session = st.session_state.all_sessions[selected_session_id]
        
        # Display session details
        conversation = selected_session['conversation_history']
        
        if conversation:
            st.markdown(f"**Session:** {selected_session['name']}")
            st.markdown(f"**Created:** {selected_session['created_at']}")
            st.markdown(f"**Total Questions:** {len(conversation)}")
            
            # Show conversation
            for i, item in enumerate(conversation, 1):
                with st.expander(f"Q{i}: {item['question'][:60]}... (Confidence: {item['confidence']:.2f})"):
                    st.markdown(f"**🕒 Time:** {item['timestamp']}")
                    st.markdown(f"**❓ Question:** {item['question']}")
                    st.markdown(f"**💡 Answer:** {item['answer']}")
                    st.markdown(f"**🎯 Confidence:** {item['confidence']:.2f}")
        else:
            st.info(f"No questions asked in {selected_session['name']} yet.")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "📄 Document Upload":
        render_document_upload()
    elif page == "❓ Ask Questions":
        render_ask_questions()
    elif page == "📊 Evaluation":
        render_evaluation()
    elif page == "📈 History":
        render_history()

if __name__ == "__main__":
    main()
