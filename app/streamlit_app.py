import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Intelligent Q&A System",
    page_icon="🤖",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []

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

def ask_question(question):
    """Ask a question to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask-question",
            json={"question": question}
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
        response = requests.get(f"{API_BASE_URL}/generate-questions/{doc_id}")
        
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
        response = requests.get(f"{API_BASE_URL}/evaluate/{doc_id}")
        
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
    init_session_state()
    
    st.title("🤖 Intelligent Q&A System")
    st.markdown("Upload documents, ask questions, and evaluate system performance")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload & Q&A", "📊 Evaluation", "📝 History"])
    
    with tab1:
        st.header("Document Upload & Question Answering")
        
        # Document Upload Section
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md']
        )
        
        if uploaded_files and st.button("Upload Documents", type="primary"):
            with st.spinner("Uploading and processing documents..."):
                result = upload_documents(uploaded_files)
                
                if result:
                    st.success(f"✅ {result['summary']['successful']} documents uploaded successfully!")
                    
                    # Store uploaded documents
                    for doc in result['documents']:
                        if doc['status'] == 'processed':
                            st.session_state.uploaded_documents.append(doc)
                    
                    # Display results
                    for doc in result['documents']:
                        if doc['status'] == 'processed':
                            st.info(f"📄 {doc['filename']}: {doc['chunks']} chunks, {doc['processing_time']}s")
                        else:
                            st.error(f"❌ {doc['filename']}: {doc.get('error', 'Unknown error')}")
        
        # Question Answering Section
        st.subheader("❓ Ask Questions")
        
        # Get current documents
        docs_data = get_documents()
        if docs_data and docs_data['documents']:
            st.info(f"📚 {docs_data['total_documents']} documents available for questioning")
        
        question = st.text_input("Enter your question:", placeholder="What would you like to know?")
        
        if question and st.button("Ask Question", type="primary"):
            with st.spinner("Generating answer..."):
                result = ask_question(question)
                
                if result:
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.write(result['answer'])
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    with col2:
                        st.metric("Response Time", f"{result['response_time']:.3f}s")
                    with col3:
                        st.metric("Sources", len(result.get('sources', [])))
                    
                    # Store in history
                    st.session_state.conversation_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
    
    with tab2:
        st.header("System Evaluation")
        
        # Get documents for evaluation
        docs_data = get_documents()
        
        if docs_data and docs_data['documents']:
            st.subheader("📋 Available Documents")
            
            # Display documents in a table
            df = pd.DataFrame(docs_data['documents'])
            st.dataframe(df, use_container_width=True)
            
            # Document selection for evaluation
            doc_options = {f"{doc['filename']} ({doc['doc_id'][:8]}...)": doc['doc_id'] 
                          for doc in docs_data['documents']}
            
            selected_doc = st.selectbox("Select document for evaluation:", list(doc_options.keys()))
            
            if selected_doc:
                doc_id = doc_options[selected_doc]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🎯 Generate Questions", type="primary"):
                        with st.spinner("Generating questions..."):
                            result = generate_questions(doc_id)
                            
                            if result:
                                st.session_state.generated_questions = result['questions']
                                st.success(f"✅ Generated {result['questions_generated']} questions!")
                                
                                # Display questions
                                st.subheader("Generated Questions:")
                                for i, q in enumerate(result['questions'], 1):
                                    st.write(f"{i}. {q}")
                
                with col2:
                    if st.button("📊 Run Evaluation", type="secondary"):
                        with st.spinner("Running evaluation..."):
                            result = run_evaluation(doc_id)
                            
                            if result and 'results' in result:
                                eval_results = result['results']
                                st.session_state.evaluation_results[doc_id] = eval_results
                                
                                st.success("✅ Evaluation completed!")
                                
                                # Display key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Average F1", f"{eval_results.get('average_f1', 0):.3f}")
                                with col2:
                                    st.metric("Semantic Score", f"{eval_results.get('average_semantic', 0):.3f}")
                                with col3:
                                    st.metric("Accuracy Rate", f"{eval_results.get('accuracy_rate', 0):.3f}")
                                with col4:
                                    st.metric("Avg Response Time", f"{eval_results.get('average_response_time', 0):.3f}s")
                                
                                # Display question types
                                if 'question_types' in eval_results:
                                    st.subheader("Question Types Generated:")
                                    types_df = pd.DataFrame(list(eval_results['question_types'].items()), 
                                                          columns=['Type', 'Count'])
                                    st.bar_chart(types_df.set_index('Type'))
                
                # Display current questions if any
                if st.session_state.generated_questions:
                    st.subheader("🎯 Current Generated Questions")
                    for i, q in enumerate(st.session_state.generated_questions, 1):
                        st.write(f"{i}. {q}")
        else:
            st.info("📤 Please upload documents first to run evaluations.")
    
    with tab3:
        st.header("Conversation History")
        
        if st.session_state.conversation_history:
            st.subheader(f"💬 {len(st.session_state.conversation_history)} Previous Interactions")
            
            # Display conversation history
            for i, interaction in enumerate(reversed(st.session_state.conversation_history), 1):
                with st.expander(f"Q{len(st.session_state.conversation_history) - i + 1}: {interaction['question'][:50]}..."):
                    st.write("**Question:**", interaction['question'])
                    st.write("**Answer:**", interaction['answer'])
                    st.write("**Confidence:**", f"{interaction['confidence']:.2f}")
                    st.write("**Time:**", interaction['timestamp'])
        else:
            st.info("💭 No conversation history yet. Ask some questions to see them here!")
        
        # Display evaluation history
        if st.session_state.evaluation_results:
            st.subheader("📊 Evaluation History")
            
            for doc_id, results in st.session_state.evaluation_results.items():
                with st.expander(f"Evaluation: {doc_id[:8]}..."):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("F1 Score", f"{results.get('average_f1', 0):.3f}")
                    with col2:
                        st.metric("Semantic Score", f"{results.get('average_semantic', 0):.3f}")
                    with col3:
                        st.metric("Accuracy", f"{results.get('accuracy_rate', 0):.3f}")

if __name__ == "__main__":
    main()
