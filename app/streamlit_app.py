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
                        st.experimental_rerun()
                    else:
                        st.error(f"❌ Error: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Q&A Section
    st.subheader("❓ Ask Questions")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    
    # Question input
    question = st.text_input("Enter your question:")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ask_button = st.button("🎯 Ask Question", type="primary", disabled=not question)
    with col2:
        if st.button("🔄 New Session"):
            st.session_state.session_id = None
            st.success("New session started!")
    
    if ask_button and question:
        with st.spinner("Generating answer..."):
            try:
                payload = {
                    "question": question,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/ask-question", 
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store session ID
                    st.session_state.session_id = result['session_id']
                    
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
            response = requests.get(f"{API_BASE_URL}/generate-questions/{doc_id}?max_questions=6", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success(f"✅ Generated {result['questions_generated']} questions!")
                
                st.subheader("📝 Generated Questions")
                for i, question in enumerate(result['questions'], 1):
                    st.write(f"{i}. {question}")
                
                # Store in session state for evaluation
                st.session_state.generated_questions = result['questions_data']
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
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_f1 = evaluation_results.get('average_f1', 0)
                    st.metric("Average F1 Score", f"{avg_f1:.3f}")
                
                with col2:
                    total_q = evaluation_results.get('total_questions', 0)
                    st.metric("Questions Tested", total_q)
                
                with col3:
                    avg_time = evaluation_results.get('average_response_time', 0)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                
                with col4:
                    errors = len(evaluation_results.get('errors', []))
                    st.metric("Errors", errors)
                
                # Performance grade
                st.subheader("🎯 Performance Assessment")
                if avg_f1 >= 0.8:
                    grade = "A+ (Excellent)"
                    color = "green"
                elif avg_f1 >= 0.7:
                    grade = "A (Very Good)"
                    color = "lightgreen"
                elif avg_f1 >= 0.6:
                    grade = "B (Good)"
                    color = "yellow"
                elif avg_f1 >= 0.5:
                    grade = "C (Acceptable)"
                    color = "orange"
                else:
                    grade = "D (Needs Improvement)"
                    color = "red"
                
                st.markdown(f"**F1 Score Grade:** :{color}[{grade}]")
                
                # Detailed results
                st.subheader("📊 Detailed Results")
                predictions = evaluation_results.get('predictions', [])
                
                if predictions:
                    for i, pred in enumerate(predictions, 1):
                        with st.expander(f"Question {i}: {pred['question'][:60]}..."):
                            st.write(f"**Question:** {pred['question']}")
                            st.write(f"**Expected:** {pred['expected']}")
                            st.write(f"**Predicted:** {pred['predicted']}")
                            st.write(f"**F1 Score:** {pred['f1_score']:.3f}")
                            st.write(f"**Response Time:** {pred['response_time']:.2f}s")
                            st.write(f"**Confidence:** {pred['confidence']:.3f}")
                
                # F1 Score distribution
                f1_scores = evaluation_results.get('f1_scores', [])
                if f1_scores:
                    st.subheader("📈 F1 Score Distribution")
                    st.bar_chart({"F1 Scores": f1_scores})
                    
                    import statistics
                    st.write(f"**Statistics:**")
                    st.write(f"- Average: {statistics.mean(f1_scores):.3f}")
                    st.write(f"- Median: {statistics.median(f1_scores):.3f}")
                    st.write(f"- Min: {min(f1_scores):.3f}")
                    st.write(f"- Max: {max(f1_scores):.3f}")
            
            else:
                st.error(f"❌ Evaluation failed: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def conversation_history():
    """Show conversation history"""
    st.header("💬 Conversation History")
    
    if not st.session_state.get('session_id'):
        st.info("No active session. Start a conversation in the Upload & Q&A tab.")
        return
    
    try:
        response = requests.get(f"{API_BASE_URL}/conversation-history/{st.session_state.session_id}", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            history = result['history']
            
            st.write(f"**Session ID:** {result['session_id']}")
            st.write(f"**Total Interactions:** {result['total_interactions']}")
            
            if history:
                for i, interaction in enumerate(reversed(history), 1):
                    with st.expander(f"Q{i}: {interaction['question'][:60]}..."):
                        st.write(f"**Question:** {interaction['question']}")
                        st.write(f"**Answer:** {interaction['answer']}")
                        st.write(f"**Confidence:** {interaction['confidence']:.3f}")
                        st.write(f"**Time:** {interaction['timestamp'][:19]}")
            else:
                st.info("No conversation history found.")
        
        else:
            st.error(f"❌ Error: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
