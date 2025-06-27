import streamlit as st
import requests
import json
from datetime import datetime
import time
import pandas as pd
import statistics

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

def upload_documents(files):
    """Upload documents to the API"""
    try:
        files_data = []
        for file in files:
            files_data.append(('files', (file.name, file.getvalue(), file.type)))
        
        response = requests.post(f"{API_BASE_URL}/upload-documents", files=files_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading documents: {str(e)}")
        return None

def ask_question(question, session_id=None):
    """Ask a question to the API"""
    try:
        payload = {
            "question": question,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask-question", 
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Question failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")
        return None

def generate_questions(doc_id):
    """Generate questions from a document"""
    try:
        # Add timestamp to ensure different questions each time
        timestamp = int(time.time() * 1000)
        response = requests.get(f"{API_BASE_URL}/generate-questions/{doc_id}?max_questions=8&seed={timestamp}", timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Question generation failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

def run_evaluation(doc_id):
    """Run evaluation on a document"""
    try:
        response = requests.get(f"{API_BASE_URL}/evaluate/{doc_id}", timeout=300)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Evaluation failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error running evaluation: {str(e)}")
        return None

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/list-documents")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get documents: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting documents: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
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

if __name__ == "__main__":
    main()
