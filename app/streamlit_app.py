import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Simple Document Q&A System",
    page_icon="🤖",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://127.0.0.1:8000"

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def main():
    st.title("🤖 Simple Document Q&A System")
    st.markdown("**Clean & Simple** - Upload documents, generate questions, and evaluate with F1 scores!")
    
    # Check backend connection
    is_connected, health_data = check_backend()
    
    if not is_connected:
        st.error("❌ Backend not connected. Please start the backend first:")
        st.code("python start.py")
        st.stop()
    
    # Show status
    st.success("✅ Backend connected")
    
    if health_data:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", health_data.get('documents_processed', 0))
        with col2:
            st.metric("Chunks", health_data.get('chunks_indexed', 0))
    
    # Main interface with three tabs
    tab1, tab2, tab3 = st.tabs([
        "📄 Upload & Q&A", 
        "🎯 Evaluation", 
        "💬 History"
    ])
    
    with tab1:
        upload_and_qa()
    
    with tab2:
        evaluation_tab()
    
    with tab3:
        conversation_history()

def upload_and_qa():
    """Upload documents and ask questions"""
    st.header("📄 Document Upload & Question Answering")
    
    # Show current documents
    st.subheader("📚 Your Documents")
    try:
        response = requests.get(f"{API_BASE_URL}/list-documents", timeout=10)
        if response.status_code == 200:
            doc_list = response.json()
            documents = doc_list.get('documents', [])
            
            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Doc ID:** {doc['doc_id']}")
                        with col2:
                            st.write(f"**Text Length:** {doc['text_length']}")
                        with col3:
                            st.write(f"**Processed:** {doc['processed_at']}")
            else:
                st.info("No documents uploaded yet.")
        else:
            st.error("Could not fetch documents")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Document upload
    st.subheader("📁 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'html', 'md'],
        help="Supported formats: PDF, Word, Text, HTML, Markdown"
    )
    
    if uploaded_files:
        if st.button("🚀 Upload & Process", type="primary"):
            with st.spinner("Processing documents..."):
                files_data = []
                for file in uploaded_files:
                    files_data.append(("files", (file.name, file.getvalue(), file.type)))
                
                try:
                    response = requests.post(f"{API_BASE_URL}/upload-documents", files=files_data, timeout=120)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success(f"✅ {result['summary']['successful']} documents processed successfully!")
                        
                        for doc in result['documents']:
                            if doc['status'] == 'processed':
                                st.success(f"✅ {doc['filename']} - {doc['chunks']} chunks created")
                            else:
                                st.error(f"❌ {doc['filename']}: {doc.get('error', 'Unknown error')}")
                        
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Q&A Section
    st.subheader("❓ Ask Questions")
    
    # Initialize session state for session management
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'all_sessions' not in st.session_state:
        st.session_state.all_sessions = []
    
    # Session management
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.current_session_id:
            st.info(f"🔗 Current Session: {st.session_state.current_session_id[:8]}...")
        else:
            st.info("🆕 No active session")
    
    with col2:
        if st.button("🔄 New Session"):
            # Save current session to history before creating new one
            if st.session_state.current_session_id and st.session_state.current_session_id not in st.session_state.all_sessions:
                st.session_state.all_sessions.append(st.session_state.current_session_id)
            
            # Clear current session
            st.session_state.current_session_id = None
            st.success("New session will be created with your next question!")
    
    # Question input
    question = st.text_input("Enter your question:")
    
    if st.button("🎯 Ask Question", type="primary", disabled=not question):
        with st.spinner("Generating answer..."):
            try:
                payload = {
                    "question": question,
                    "session_id": st.session_state.current_session_id
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/ask-question", 
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Update current session ID
                    st.session_state.current_session_id = result['session_id']
                    
                    # Add to all sessions if not already there
                    if result['session_id'] not in st.session_state.all_sessions:
                        st.session_state.all_sessions.append(result['session_id'])
                    
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(result['answer'])
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    with col2:
                        st.metric("Response Time", f"{result['response_time']:.2f}s")
                    with col3:
                        st.metric("Sources", len(result.get('sources', [])))
                    
                    # Display sources
                    if result.get('sources'):
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i}: {source['source']}"):
                                st.write(f"**Similarity:** {source['similarity']:.3f}")
                                st.write(f"**Preview:** {source.get('preview', 'No preview')}")
                
                else:
                    st.error(f"❌ Error: {response.text}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def evaluation_tab():
    """Evaluation interface"""
    st.header("🎯 F1 Score Evaluation")
    
    # Get documents for evaluation
    try:
        response = requests.get(f"{API_BASE_URL}/list-documents", timeout=10)
        if response.status_code == 200:
            doc_list = response.json()
            documents = doc_list.get('documents', [])
            
            if not documents:
                st.warning("⚠️ No documents available. Please upload documents first.")
                return
            
            # Document selector
            doc_options = {f"{doc['filename']} ({doc['chunk_count']} chunks)": doc['doc_id'] 
                          for doc in documents}
            
            selected_doc_name = st.selectbox(
                "Select document for evaluation:",
                list(doc_options.keys())
            )
            
            if selected_doc_name:
                selected_doc_id = doc_options[selected_doc_name]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📝 Generate Questions", type="secondary"):
                        generate_questions(selected_doc_id)
                
                with col2:
                    if st.button("🎯 Run F1 Evaluation", type="primary"):
                        run_evaluation(selected_doc_id)
        
        else:
            st.error("Could not fetch documents")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

def generate_questions(doc_id: str):
    """Generate questions from document"""
    with st.spinner("Generating questions from document..."):
        try:
            response = requests.get(f"{API_BASE_URL}/generate-questions/{doc_id}?max_questions=8", timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success(f"✅ Generated {result['questions_generated']} questions!")
                
                st.subheader("📝 Generated Questions")
                questions_data = result.get('questions_data', [])
                
                if questions_data:
                    for i, q_data in enumerate(questions_data, 1):
                        question = q_data.get('question', 'No question')
                        question_type = q_data.get('question_type', 'general')
                        st.write(f"{i}. **[{question_type.title()}]** {question}")
                else:
                    # Fallback to simple list
                    for i, question in enumerate(result.get('questions', []), 1):
                        st.write(f"{i}. {question}")
                
                # Store in session state for evaluation
                st.session_state.generated_questions = questions_data
                st.session_state.source_doc_id = doc_id
            
            else:
                st.error(f"❌ Error: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def run_evaluation(doc_id: str):
    """Run F1 evaluation"""
    with st.spinner("Running F1 evaluation... This may take a few minutes."):
        try:
            response = requests.get(f"{API_BASE_URL}/evaluate/{doc_id}", timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                evaluation_results = result['results']
                
                st.success("✅ F1 Evaluation completed!")
                
                # Display summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    avg_f1 = evaluation_results.get('average_f1', 0)
                    st.metric("F1 Score", f"{avg_f1:.3f}")

                with col2:
                    avg_semantic = evaluation_results.get('average_semantic', 0)
                    st.metric("Semantic Score", f"{avg_semantic:.3f}")

                with col3:
                    accuracy_rate = evaluation_results.get('accuracy_rate', 0)
                    st.metric("Accuracy Rate", f"{accuracy_rate:.3f}")

                with col4:
                    total_q = evaluation_results.get('total_questions', 0)
                    st.metric("Questions", total_q)

                with col5:
                    avg_time = evaluation_results.get('average_response_time', 0)
                    st.metric("Avg Time", f"{avg_time:.2f}s")

                # Show question types breakdown
                question_types = evaluation_results.get('question_types', {})
                if question_types:
                    st.subheader("📊 Question Types Generated")
                    type_cols = st.columns(len(question_types))
                    for i, (q_type, count) in enumerate(question_types.items()):
                        with type_cols[i % len(type_cols)]:
                            st.metric(q_type.title(), count)

                # Performance grade
                st.subheader("🎯 Performance Assessment")
                semantic_score = evaluation_results.get('average_semantic', 0)
                if semantic_score >= 0.9:
                    grade = "A+ (Excellent)"
                    color = "green"
                elif semantic_score >= 0.8:
                    grade = "A (Very Good)"
                    color = "lightgreen"
                elif semantic_score >= 0.7:
                    grade = "B (Good)"
                    color = "yellow"
                elif semantic_score >= 0.6:
                    grade = "C (Acceptable)"
                    color = "orange"
                else:
                    grade = "D (Needs Improvement)"
                    color = "red"

                st.markdown(f"**Overall Grade (Semantic):** :{color}[{grade}]")
                
                # Detailed results
                st.subheader("📊 Detailed Results")
                predictions = evaluation_results.get('predictions', [])
                
                if predictions:
                    for i, pred in enumerate(predictions, 1):
                        question_type = pred.get('question_type', 'general')
                        with st.expander(f"Q{i} [{question_type.title()}]: {pred['question'][:60]}..."):
                            st.write(f"**Question:** {pred['question']}")
                            st.write(f"**Expected:** {pred['expected']}")
                            st.write(f"**Predicted:** {pred['predicted']}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**F1 Score:** {pred['f1_score']:.3f}")
                            with col2:
                                st.write(f"**Semantic:** {pred['semantic_score']:.3f}")
                            with col3:
                                contains = "✅ Yes" if pred['contains_answer'] else "❌ No"
                                st.write(f"**Contains Answer:** {contains}")
                            
                            st.write(f"**Response Time:** {pred['response_time']:.2f}s")
                            st.write(f"**Confidence:** {pred['confidence']:.3f}")
                
                # Score distributions
                f1_scores = evaluation_results.get('f1_scores', [])
                semantic_scores = evaluation_results.get('semantic_scores', [])
                
                if f1_scores and semantic_scores:
                    st.subheader("📈 Score Distributions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**F1 Scores**")
                        st.bar_chart({"F1 Scores": f1_scores})
                        
                        import statistics
                        st.write(f"- Average: {statistics.mean(f1_scores):.3f}")
                        st.write(f"- Min: {min(f1_scores):.3f}")
                        st.write(f"- Max: {max(f1_scores):.3f}")
                    
                    with col2:
                        st.write("**Semantic Scores**")
                        st.bar_chart({"Semantic Scores": semantic_scores})
                        
                        st.write(f"- Average: {statistics.mean(semantic_scores):.3f}")
                        st.write(f"- Min: {min(semantic_scores):.3f}")
                        st.write(f"- Max: {max(semantic_scores):.3f}")
            
            else:
                st.error(f"❌ Evaluation failed: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def conversation_history():
    """Show conversation history with improved session management"""
    st.header("💬 Conversation History")
    
    # Initialize session state if not exists
    if 'all_sessions' not in st.session_state:
        st.session_state.all_sessions = []
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    # Add current session to all sessions if it exists and not already there
    if (st.session_state.current_session_id and 
        st.session_state.current_session_id not in st.session_state.all_sessions):
        st.session_state.all_sessions.append(st.session_state.current_session_id)
    
    # Show all available sessions
    all_sessions = st.session_state.all_sessions.copy()
    if st.session_state.current_session_id and st.session_state.current_session_id not in all_sessions:
        all_sessions.append(st.session_state.current_session_id)
    
    if not all_sessions:
        st.info("💬 No conversation sessions found. Start a conversation in the Upload & Q&A tab to see history.")
        return
    
    # Session selector
    st.subheader("📋 Select Session to View")
    
    # Create session options with labels
    session_options = {}
    for session_id in all_sessions:
        if session_id == st.session_state.current_session_id:
            label = f"🔴 Current Session: {session_id[:8]}..."
        else:
            label = f"📝 Session: {session_id[:8]}..."
        session_options[label] = session_id
    
    # Default to current session if available
    default_session = None
    if st.session_state.current_session_id:
        for label, session_id in session_options.items():
            if session_id == st.session_state.current_session_id:
                default_session = label
                break
    
    # Session selector
    if session_options:
        selected_session_label = st.selectbox(
            "Choose a session:",
            list(session_options.keys()),
            index=list(session_options.keys()).index(default_session) if default_session else 0
        )
        
        selected_session_id = session_options[selected_session_label]
        
        # Fetch and display conversation history
        try:
            response = requests.get(f"{API_BASE_URL}/conversation-history/{selected_session_id}", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                history = result['history']
                
                # Session info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Session ID", f"{result['session_id'][:8]}...")
                with col2:
                    st.metric("Total Interactions", result['total_interactions'])
                with col3:
                    status = "🔴 Active" if selected_session_id == st.session_state.current_session_id else "📝 Archived"
                    st.metric("Status", status)
                
                if history:
                    st.subheader("💬 Conversation")
                    
                    # Display conversations in chronological order (most recent first)
                    for i, interaction in enumerate(reversed(history), 1):
                        with st.expander(f"Q{i}: {interaction['question'][:60]}..."):
                            st.write(f"**Question:** {interaction['question']}")
                            st.write(f"**Answer:** {interaction['answer']}")
                            
                            # Metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Confidence:** {interaction['confidence']:.3f}")
                            with col2:
                                st.write(f"**Time:** {interaction['timestamp'][:19]}")
                else:
                    st.info("📝 No interactions found in this session.")
            
            else:
                st.error(f"❌ Error fetching session history: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Session management buttons
    st.divider()
    st.subheader("🛠️ Session Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Sessions"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All History") and st.session_state.all_sessions:
            if st.button("⚠️ Confirm Clear All", type="secondary"):
                st.session_state.all_sessions = []
                st.session_state.current_session_id = None
                st.success("All session history cleared!")
                st.rerun()
    
    with col3:
        if st.session_state.current_session_id:
            st.info(f"Current: {st.session_state.current_session_id[:8]}...")
        else:
            st.info("No active session")

if __name__ == "__main__":
    main()
