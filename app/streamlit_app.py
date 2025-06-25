import streamlit as st
import requests
import json
from datetime import datetime
import time

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly not installed. Charts will be simplified. Install with: pip install plotly pandas")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Document Q&A System with Evaluation",
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
    st.title("🤖 Enhanced Document Q&A System with Evaluation")
    st.markdown("**Version 2.0** - Now with SQUAD 2.0 and COQA evaluation capabilities!")
    
    # Check backend connection
    is_connected, health_data = check_backend()
    
    if not is_connected:
        st.error("❌ Backend not connected. Please start the backend first:")
        st.code("python start.py")
        st.stop()
    
    # Show enhanced status
    st.success("✅ Backend connected")
    
    if health_data:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Documents", health_data.get('documents_processed', 0))
        with col2:
            st.metric("Chunks", health_data.get('chunks_indexed', 0))
        with col3:
            st.metric("Version", health_data.get('version', '2.0.0'))
        with col4:
            st.metric("Evaluation", "✅ Ready" if health_data.get('evaluation_ready') else "❌ Not Ready")
        with col5:
            config = health_data.get('config', {})
            st.metric("Max Answer Length", config.get('max_answer_length', 2000))
    
    # Main interface with three tabs as per user request
    tab1, tab2, tab3 = st.tabs([
        "📄 Q&A", 
        "💬 History", 
        "📈 Performance Metrics"
    ])
    
    with tab1:
        upload_and_qa()
    
    with tab2:
        conversation_history()
    
    with tab3:
        performance_metrics()

def performance_metrics():
    """Combined Performance Metrics tab merging evaluation, benchmark, and system stats"""
    st.header("📈 Performance Metrics")
    
    # Show system stats
    try:
        response = requests.get(f"{API_BASE_URL}/system-stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            
            # Evaluation readiness section
            st.subheader("🎯 Evaluation Readiness")
            eval_readiness = stats.get('evaluation_readiness', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                docs_indexed = eval_readiness.get('documents_indexed', False)
                st.metric("Documents Indexed", "✅ Yes" if docs_indexed else "❌ No")
            
            with col2:
                eval_ready = eval_readiness.get('evaluation_system_ready', False)
                st.metric("Evaluation System", "✅ Ready" if eval_ready else "❌ Not Ready")
            
            with col3:
                prod_ready = eval_readiness.get('ready_for_production', False)
                st.metric("Production Ready", "✅ Yes" if prod_ready else "❌ No")
            
            with col4:
                supported_evals = eval_readiness.get('supported_evaluations', [])
                st.metric("Supported Evaluations", len(supported_evals))
            
            if supported_evals:
                st.write("**Supported Evaluation Types:**", ", ".join(supported_evals))
            
            # Document statistics
            st.subheader("📄 Document Statistics")
            doc_stats = stats.get('documents', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Processed", doc_stats.get('total_processed', 0))
            with col2:
                st.metric("Total Chunks", doc_stats.get('total_chunks', 0))
            with col3:
                st.metric("Total Text Length", doc_stats.get('total_text_length', 0))
            
            # Session statistics
            st.subheader("💬 Session Statistics")
            session_stats = stats.get('sessions', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Sessions", session_stats.get('active_sessions', 0))
            with col2:
                st.metric("Total Interactions", session_stats.get('total_interactions', 0))
            
            # System configuration
            st.subheader("⚙️ System Configuration")
            config = stats.get('configuration', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Chunk Size", config.get('max_chunk_size', 0))
            with col2:
                st.metric("Max Retrieval Chunks", config.get('max_retrieval_chunks', 0))
            with col3:
                st.metric("Max Answer Length", config.get('max_answer_length', 0))
            with col4:
                st.write(f"**Model**: {config.get('generation_model', 'Unknown')}")
            
            # Cache statistics
            st.subheader("🗄️ Cache Statistics")
            cache_stats = stats.get('cache', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Query Cache Size", cache_stats.get('query_cache_size', 0))
            with col2:
                st.metric("Embedding Cache Size", cache_stats.get('embedding_cache_size', 0))
        else:
            st.error(f"❌ Error fetching system stats: {response.text}")
    except Exception as e:
        st.error(f"❌ Error fetching system stats: {str(e)}")
    
    st.divider()
    
    # Evaluation interface
    st.subheader("📊 System Evaluation")
    
    try:
        stats_response = requests.get(f"{API_BASE_URL}/system-stats", timeout=10)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            eval_readiness = stats.get('evaluation_readiness', {})
            
            if not eval_readiness.get('documents_indexed', False):
                st.warning("⚠️ No documents indexed. Please upload documents or add sample content first.")
                
                return
            
            st.success("✅ System ready for evaluation!")
            
        else:
            st.error("❌ Could not check system readiness")
            return
            
    except Exception as e:
        st.error(f"❌ Error checking system status: {str(e)}")
        return
    
    # Evaluation controls
    st.subheader("🚀 Run Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Run SQUAD Evaluation", type="primary"):
            run_evaluation("squad")
    
    with col2:
        if st.button("📊 Get Evaluation Report", type="secondary"):
            get_evaluation_report()

def run_evaluation(eval_type: str):
    """Run system evaluation"""
    with st.spinner(f"Running {eval_type.upper()} evaluation... This may take a few minutes."):
        try:
            # First check if we have documents
            stats_response = requests.get(f"{API_BASE_URL}/system-stats", timeout=10)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                doc_count = stats.get('documents', {}).get('total_processed', 0)
                if doc_count == 0:
                    st.error("❌ No documents found! Please upload documents first.")
                    return
            
            payload = {
                "dataset_type": eval_type,
                "include_performance": True
            }
            
            response = requests.post(
                f"{API_BASE_URL}/evaluate", 
                json=payload, 
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Evaluation completed!")
                
                # Check if there was an error in the results
                if "error" in result.get("results", {}):
                    st.error(f"❌ Evaluation error: {result['results']['error']}")
                    return
                
                results_data = result.get('results', {})
                
                # Display summary metrics
                if 'squad_results' in results_data:
                    squad_results = results_data['squad_results']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        f1_score = squad_results.get('average_f1', 0)
                        st.metric("F1 Score", f"{f1_score:.3f}", 
                                 delta=f"{f1_score - 0.75:.3f}" if f1_score > 0 else None)
                    
                    with col2:
                        em_score = squad_results.get('exact_match_score', 0)
                        st.metric("Exact Match", f"{em_score:.3f}",
                                 delta=f"{em_score - 0.65:.3f}" if em_score > 0 else None)
                    
                    with col3:
                        response_time = squad_results.get('average_response_time', 0)
                        st.metric("Avg Response Time", f"{response_time:.2f}s",
                                 delta=f"{3.0 - response_time:.2f}s")
                    
                    with col4:
                        error_count = len(squad_results.get('errors', []))
                        total_questions = squad_results.get('total_questions', 1)
                        error_rate = error_count / total_questions
                        st.metric("Error Rate", f"{error_rate:.3f}",
                                 delta=f"{0.01 - error_rate:.3f}")
                
                # Detailed results
                st.subheader("📈 Detailed Results")
                
                # SQUAD results
                if 'squad_results' in results_data:
                    squad_results = results_data['squad_results']
                    
                    st.write("**SQUAD 2.0 Results:**")
                    squad_col1, squad_col2, squad_col3 = st.columns(3)
                    
                    with squad_col1:
                        st.write(f"Total Questions: {squad_results.get('total_questions', 0)}")
                        st.write(f"Exact Matches: {squad_results.get('exact_matches', 0)}")
                    
                    with squad_col2:
                        st.write(f"Average F1: {squad_results.get('average_f1', 0):.3f}")
                        st.write(f"Response Time: {squad_results.get('average_response_time', 0):.3f}s")
                    
                    with squad_col3:
                        st.write(f"Errors: {len(squad_results.get('errors', []))}")
                    
                    # Show individual predictions if available
                    predictions = squad_results.get('predictions', [])
                    if predictions:
                        st.write("**Sample Predictions:**")
                        for i, pred in enumerate(predictions[:3]):  # Show first 3
                            with st.expander(f"Question {i+1}: {pred['question'][:60]}..."):
                                st.write(f"**Question:** {pred['question']}")
                                st.write(f"**Predicted:** {pred['predicted']}")
                                st.write(f"**Expected:** {pred['expected']}")
                                st.write(f"**F1 Score:** {pred['f1_score']:.3f}")
                                st.write(f"**Exact Match:** {'✅ Yes' if pred['exact_match'] else '❌ No'}")
                                st.write(f"**Confidence:** {pred.get('confidence', 0):.3f}")
                    
                    # F1 Score distribution
                    f1_scores = squad_results.get('f1_scores', [])
                    if f1_scores and len(f1_scores) > 1:
                        st.write("**F1 Score Distribution:**")
                        if PLOTLY_AVAILABLE:
                            fig = px.histogram(
                                x=f1_scores, 
                                nbins=min(10, len(f1_scores)), 
                                title="F1 Score Distribution",
                                labels={'x': 'F1 Score', 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to simple statistics
                            import statistics
                            st.write(f"- Average F1: {statistics.mean(f1_scores):.3f}")
                            st.write(f"- Median F1: {statistics.median(f1_scores):.3f}")
                            st.write(f"- Min F1: {min(f1_scores):.3f}")
                            st.write(f"- Max F1: {max(f1_scores):.3f}")
                            st.bar_chart({"F1 Scores": f1_scores[:10]})  # Show first 10 as simple bar chart
                
                # Performance results
                if 'performance_results' in results_data:
                    perf_results = results_data['performance_results']
                    
                    st.write("**Performance Results:**")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.write(f"Avg Response Time: {perf_results.get('average_response_time', 0):.3f}s")
                        st.write(f"P95 Response Time: {perf_results.get('p95_response_time', 0):.3f}s")
                    
                    with perf_col2:
                        st.write(f"Throughput: {perf_results.get('throughput', 0):.2f} req/s")
                        st.write(f"Error Rate: {perf_results.get('error_rate', 0):.3f}")
                    
                    with perf_col3:
                        st.write(f"Memory Usage: {perf_results.get('average_memory_usage', 0):.2f} MB")
                
                # Performance grading
                st.subheader("🎯 Performance Assessment")
                
                if 'squad_results' in results_data:
                    squad_results = results_data['squad_results']
                    f1_score = squad_results.get('average_f1', 0)
                    response_time = squad_results.get('average_response_time', 0)
                    
                    grade_col1, grade_col2 = st.columns(2)
                    
                    with grade_col1:
                        # Accuracy grade
                        if f1_score >= 0.8:
                            accuracy_grade = "A+ (Excellent)"
                            accuracy_color = "green"
                        elif f1_score >= 0.75:
                            accuracy_grade = "A (Very Good)"
                            accuracy_color = "lightgreen"
                        elif f1_score >= 0.7:
                            accuracy_grade = "B (Good)"
                            accuracy_color = "yellow"
                        elif f1_score >= 0.6:
                            accuracy_grade = "C (Acceptable)"
                            accuracy_color = "orange"
                        else:
                            accuracy_grade = "D (Needs Improvement)"
                            accuracy_color = "red"
                        
                        st.markdown(f"**Accuracy Grade:** :{accuracy_color}[{accuracy_grade}]")
                    
                    with grade_col2:
                        # Performance grade
                        if response_time < 2.0:
                            perf_grade = "A+ (Excellent)"
                            perf_color = "green"
                        elif response_time < 3.0:
                            perf_grade = "A (Very Good)"
                            perf_color = "lightgreen"
                        elif response_time < 5.0:
                            perf_grade = "B (Good)"
                            perf_color = "yellow"
                        elif response_time < 8.0:
                            perf_grade = "C (Acceptable)"
                            perf_color = "orange"
                        else:
                            perf_grade = "D (Needs Improvement)"
                            perf_color = "red"
                        
                        st.markdown(f"**Performance Grade:** :{perf_color}[{perf_grade}]")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    recommendations = []
                    
                    if f1_score < 0.75:
                        recommendations.append("🎯 **Improve Accuracy**: Consider fine-tuning retrieval parameters or improving document chunking")
                    
                    if response_time > 5.0:
                        recommendations.append("⚡ **Optimize Performance**: Implement caching or optimize embedding generation")
                    
                    error_count = len(squad_results.get('errors', []))
                    total_questions = squad_results.get('total_questions', 1)
                    if error_count / total_questions > 0.05:
                        recommendations.append("🛠️ **Reduce Errors**: Improve error handling and input validation")
                    
                    if not recommendations:
                        recommendations.append("🎉 **Excellent Performance**: System meets production-quality standards!")
                    
                    for rec in recommendations:
                        st.write(rec)
                
                # Show document used for evaluation
                doc_used = result.get('document_used')
                if doc_used:
                    st.info(f"📄 Evaluation performed using document: {doc_used}")
            
            else:
                error_detail = ""
                try:
                    error_response = response.json()
                    error_detail = error_response.get('detail', 'Unknown error')
                except:
                    error_detail = response.text
                
                st.error(f"❌ Evaluation failed: {error_detail}")
                
                # Show debugging information
                st.write("**Debug Information:**")
                st.write(f"- Status Code: {response.status_code}")
                st.write(f"- Response: {response.text[:500]}")
        
        except requests.exceptions.Timeout:
            st.error("❌ Evaluation timed out. The system might be processing large documents.")
        except Exception as e:
            st.error(f"❌ Error running evaluation: {str(e)}")
            import traceback
            st.write("**Error Details:**")
            st.code(traceback.format_exc())

def get_evaluation_report():
    """Get detailed evaluation report"""
    with st.spinner("Generating comprehensive evaluation report..."):
        try:
            response = requests.get(f"{API_BASE_URL}/evaluation-report", timeout=120)
            
            if response.status_code == 200:
                report = response.text
                
                st.success("✅ Evaluation report generated!")
                
                # Display report
                st.subheader("📋 Comprehensive Evaluation Report")
                st.text_area("Report Content", report, height=400)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"qa_system_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            else:
                st.error(f"❌ Failed to generate report: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")

def upload_and_qa():
    """Enhanced upload and Q&A interface"""
    st.header("📄 Document Upload & Question Answering")
    
    # Debug section - show current document status
    st.subheader("🔍 Current System Status")
    try:
        response = requests.get(f"{API_BASE_URL}/system-stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            debug_info = stats.get('debug_info', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Processed", debug_info.get('processed_document_count', 0))
            with col2:
                st.metric("Chunks Indexed", debug_info.get('chunk_embeddings_count', 0))
            with col3:
                doc_ids = debug_info.get('processed_document_ids', [])
                st.metric("Document IDs", len(doc_ids))
            
            if doc_ids:
                st.success(f"✅ Documents found: {', '.join(doc_ids[:3])}{'...' if len(doc_ids) > 3 else ''}")
            else:
                st.warning("⚠️ No documents currently indexed")
        else:
            st.error("❌ Could not fetch system status")
    except Exception as e:
        st.error(f"❌ Error checking status: {str(e)}")
    
    st.divider()
    
    # Document upload
    st.subheader("📁 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload and process",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'html', 'md'],
        help="Supported formats: PDF, Word documents, text files, HTML, and Markdown"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                files_data = []
                for file in uploaded_files:
                    files_data.append(("files", (file.name, file.getvalue(), file.type)))
                
                try:
                    response = requests.post(f"{API_BASE_URL}/upload-documents", files=files_data, timeout=120)
                    
                    if response.status_code == 200:
                        result = response.json()
                        summary = result.get('summary', {})
                        
                        # Show summary with evaluation readiness
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Files", summary.get('total_files', 0))
                        with col2:
                            st.metric("Successful", summary.get('successful', 0))
                        with col3:
                            st.metric("Failed", summary.get('failed', 0))
                        with col4:
                            st.metric("Total Chunks", summary.get('total_chunks_created', 0))
                        with col5:
                            eval_ready = summary.get('evaluation_ready_docs', 0)
                            st.metric("Eval Ready", eval_ready)
                        
                        if summary.get('ready_for_evaluation'):
                            st.success("🎯 System is ready for evaluation!")
                        
                        # Show detailed results
                        st.subheader("📋 Processing Results")
                        for doc in result['documents']:
                            if doc['status'] == 'processed':
                                eval_status = "✅ Ready" if doc.get('evaluation_ready') else "⚠️ Limited"
                                st.success(f"✅ **{doc['filename']}** - Evaluation: {eval_status}")
                                st.info(f"📄 Document ID: {doc['doc_id']}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.write(f"Chunks: {doc['chunks']}")
                                with col2:
                                    st.write(f"Text Length: {doc.get('text_length', 0)}")
                                with col3:
                                    st.write(f"Time: {doc.get('processing_time', 0):.2f}s")
                                with col4:
                                    st.write(f"Eval Ready: {'Yes' if doc.get('evaluation_ready') else 'No'}")
                            else:
                                st.error(f"❌ **{doc['filename']}**: {doc.get('error', 'Unknown error')}")
                        
                        # Force refresh the page to update status
                        st.success("✅ Upload complete! The system status above should now show your documents.")
                        if st.button("🔄 Refresh Status"):
                            st.experimental_rerun()
                    
                    else:
                        st.error(f"❌ Error: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Add a manual refresh button
    if st.button("🔄 Refresh Document Status"):
        st.experimental_rerun()
    
    st.divider()
    
    # Enhanced Q&A section
    st.subheader("❓ Ask Questions")
    
    # Sample questions for evaluation
    st.write("**Try these evaluation-focused questions:**")
    sample_questions = [
        "What evaluation datasets are recommended for Q&A systems?",
        "How is F1 score calculated for question answering?",
        "What are the performance targets for the system?",
        "Explain the difference between SQUAD and COQA evaluation",
        "What metrics should be used to assess conversational coherence?",
        "How should the system handle unanswerable questions?",
        "What are the key components of a production-ready evaluation pipeline?"
    ]
    
    selected_question = st.selectbox("Choose a sample question:", [""] + sample_questions)
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    
    # Question input
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Enter your question:", value=selected_question)
    with col2:
        if st.button("🔄 New Session"):
            st.session_state.session_id = None
            st.success("New session started!")
    
    if question:
        if st.button("🎯 Ask Question", type="primary"):
            with st.spinner("Generating comprehensive answer..."):
                try:
                    payload = {
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "context_limit": 5
                    }
                    
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(
                        f"{API_BASE_URL}/ask-question", 
                        json=payload, 
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store session ID
                        st.session_state.session_id = result['session_id']
                        
                        # Display enhanced answer
                        st.subheader("💡 Answer")
                        st.write(result['answer'])
                        
                        # Display enhanced metrics with evaluation grades
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            confidence = result['confidence']
                            st.metric("Confidence", f"{confidence:.2f}")
                        
                        with col2:
                            response_time = result.get('response_time', 0)
                            st.metric("Response Time", f"{response_time:.2f}s")
                        
                        with col3:
                            st.metric("Sources Used", result.get('source_count', 0))
                        
                        with col4:
                            answer_quality = result.get('answer_quality', {})
                            length_score = answer_quality.get('length_score', 0)
                            st.metric("Answer Quality", f"{length_score:.2f}")
                        
                        with col5:
                            st.metric("Chunks Retrieved", result.get('retrieved_chunks', 0))
                        
                        # Show evaluation grades
                        eval_metrics = result.get('evaluation_metrics', {})
                        if eval_metrics:
                            st.subheader("📊 Evaluation Metrics")
                            
                            grade_col1, grade_col2 = st.columns(2)
                            with grade_col1:
                                response_grade = eval_metrics.get('response_time_grade', 'C')
                                grade_color = 'green' if response_grade == 'A' else 'yellow' if response_grade == 'B' else 'red'
                                st.markdown(f"**Response Time Grade:** :{grade_color}[{response_grade}]")
                            
                            with grade_col2:
                                confidence_grade = eval_metrics.get('confidence_grade', 'C')
                                grade_color = 'green' if confidence_grade == 'A' else 'yellow' if confidence_grade == 'B' else 'red'
                                st.markdown(f"**Confidence Grade:** :{grade_color}[{confidence_grade}]")
                        
                        # Display detailed sources
                        if result['sources']:
                            st.subheader("📚 Detailed Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(f"Source {i}: {source['source']} (Similarity: {source['similarity']:.3f})"):
                                    st.write(f"**Cosine Similarity**: {source.get('cosine_similarity', 0):.3f}")
                                    st.write(f"**Keyword Overlap**: {source.get('keyword_overlap', 0):.3f}")
                                    st.write(f"**Preview**: {source.get('preview', 'No preview available')}")
                    
                    else:
                        st.error(f"❌ Error: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def conversation_history():
    """Show conversation history with evaluation metrics"""
    st.header("💬 Conversation History & Metrics")
    
    if not st.session_state.get('session_id'):
        st.info("No active session. Start a conversation to see history and metrics.")
        return
    
    try:
        response = requests.get(f"{API_BASE_URL}/conversation-history/{st.session_state.session_id}", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            history = result['history']
            metrics = result.get('conversation_metrics', {})
            
            # Display conversation metrics
            st.subheader("📊 Conversation Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Interactions", metrics.get('total_interactions', 0))
            with col2:
                avg_conf = metrics.get('average_confidence', 0)
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            with col3:
                avg_length = metrics.get('average_answer_length', 0)
                st.metric("Avg Answer Length", f"{avg_length:.0f}")
            with col4:
                quality = metrics.get('conversation_quality', 'Unknown')
                quality_color = 'green' if quality == 'High' else 'yellow' if quality == 'Medium' else 'red'
                st.markdown(f"**Quality:** :{quality_color}[{quality}]")
            
            st.write(f"**Session:** {result['session_id']}")
            
            if history:
                st.subheader("💬 Conversation History")
                for i, interaction in enumerate(reversed(history), 1):
                    with st.expander(f"Q{i}: {interaction['question'][:80]}..."):
                        st.write("**Question:**", interaction['question'])
                        st.write("**Answer:**", interaction['answer'])
                        
                        # Show metrics for each interaction
                        conf = interaction.get('confidence', 0)
                        conf_color = 'green' if conf > 0.8 else 'yellow' if conf > 0.6 else 'red'
                        st.markdown(f"**Confidence:** :{conf_color}[{conf:.3f}]")
                        st.write("**Time:**", interaction['timestamp'][:19])
            else:
                st.info("No history found.")
        
        else:
            st.error(f"❌ Error: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def system_stats():
    """Enhanced system statistics with evaluation readiness"""
    st.header("📈 System Statistics & Evaluation Readiness")
    
    try:
        response = requests.get(f"{API_BASE_URL}/system-stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            
            # Evaluation readiness section
            st.subheader("🎯 Evaluation Readiness")
            eval_readiness = stats.get('evaluation_readiness', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                docs_indexed = eval_readiness.get('documents_indexed', False)
                st.metric("Documents Indexed", "✅ Yes" if docs_indexed else "❌ No")
            
            with col2:
                eval_ready = eval_readiness.get('evaluation_system_ready', False)
                st.metric("Evaluation System", "✅ Ready" if eval_ready else "❌ Not Ready")
            
            with col3:
                prod_ready = eval_readiness.get('ready_for_production', False)
                st.metric("Production Ready", "✅ Yes" if prod_ready else "❌ No")
            
            with col4:
                supported_evals = eval_readiness.get('supported_evaluations', [])
                st.metric("Supported Evaluations", len(supported_evals))
            
            if supported_evals:
                st.write("**Supported Evaluation Types:**", ", ".join(supported_evals))
            
            # Document statistics
            st.subheader("📄 Document Statistics")
            doc_stats = stats.get('documents', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Processed", doc_stats.get('total_processed', 0))
            with col2:
                st.metric("Total Chunks", doc_stats.get('total_chunks', 0))
            with col3:
                st.metric("Total Text Length", doc_stats.get('total_text_length', 0))
            
            # Session statistics
            st.subheader("💬 Session Statistics")
            session_stats = stats.get('sessions', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Sessions", session_stats.get('active_sessions', 0))
            with col2:
                st.metric("Total Interactions", session_stats.get('total_interactions', 0))
            
            # System configuration
            st.subheader("⚙️ System Configuration")
            config = stats.get('configuration', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Chunk Size", config.get('max_chunk_size', 0))
            with col2:
                st.metric("Max Retrieval Chunks", config.get('max_retrieval_chunks', 0))
            with col3:
                st.metric("Max Answer Length", config.get('max_answer_length', 0))
            with col4:
                st.write(f"**Model**: {config.get('generation_model', 'Unknown')}")
            
            # Cache statistics
            st.subheader("🗄️ Cache Statistics")
            cache_stats = stats.get('cache', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Query Cache Size", cache_stats.get('query_cache_size', 0))
            with col2:
                st.metric("Embedding Cache Size", cache_stats.get('embedding_cache_size', 0))
        
        else:
            st.error(f"❌ Error: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
