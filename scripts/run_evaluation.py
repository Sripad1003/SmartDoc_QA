import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import DocumentProcessor
from app.rag_system import RAGSystem
from app.evaluation_system import EvaluationSystem
from app.config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run SQUAD evaluation and report results"""
    try:
        print("🚀 Starting SQUAD Evaluation System...")
        
        # Initialize components
        doc_processor = DocumentProcessor()
        rag_system = RAGSystem()
        evaluation_system = EvaluationSystem(rag_system, doc_processor)
        
        print("✅ Components initialized successfully")
        
        # Check if documents are already uploaded
        if not doc_processor.processed_documents:
            print("❌ No documents found!")
            print("📄 Please upload documents first using the web interface or API")
            print("   1. Start the system: python start.py")
            print("   2. Upload documents via http://127.0.0.1:8501")
            print("   3. Then run this evaluation script")
            return None
        
        print(f"📄 Found {len(doc_processor.processed_documents)} processed documents")
        
        # Get all chunks from existing documents
        all_chunks = doc_processor.get_all_chunks()
        if not all_chunks:
            print("❌ No chunks found in processed documents!")
            return None
            
        print(f"📊 Found {len(all_chunks)} chunks total")
        
        # Index documents if not already indexed
        if not rag_system.chunk_embeddings:
            print("🔗 Indexing documents with embeddings...")
            await rag_system.index_documents(all_chunks)
        
        print(f"✅ Indexed {len(rag_system.chunk_embeddings)} chunks")
        
        # Run comprehensive evaluation
        print("🧪 Running comprehensive SQUAD evaluation...")
        results = await evaluation_system.run_comprehensive_evaluation()
        
        # Extract and display results
        print("\n" + "="*60)
        print("📊 SQUAD EVALUATION RESULTS")
        print("="*60)
        
        if "squad_results" in results:
            squad = results["squad_results"]
            print(f"📋 Total Questions Evaluated: {squad.get('total_questions', 0)}")
            print(f"🎯 Exact Match Score: {squad.get('exact_match_score', 0):.3f}")
            print(f"📈 F1 Score: {squad.get('average_f1', 0):.3f}")
            print(f"⏱️  Average Response Time: {squad.get('average_response_time', 0):.3f}s")
            print(f"❌ Errors: {len(squad.get('errors', []))}")
            
            # Performance assessment
            f1_score = squad.get('average_f1', 0)
            em_score = squad.get('exact_match_score', 0)
            
            print(f"\n🎯 ACCURACY ASSESSMENT:")
            if f1_score >= 0.8:
                print("   F1 Score: A+ (Excellent) - Exceeds target of 0.75")
            elif f1_score >= 0.75:
                print("   F1 Score: A (Very Good) - Meets target of 0.75")
            elif f1_score >= 0.7:
                print("   F1 Score: B (Good) - Close to target")
            else:
                print("   F1 Score: C (Needs Improvement) - Below target")
            
            if em_score >= 0.65:
                print("   Exact Match: A (Excellent) - Meets target of 0.65")
            elif em_score >= 0.6:
                print("   Exact Match: B (Good) - Close to target")
            else:
                print("   Exact Match: C (Needs Improvement) - Below target")
        
        if "performance_results" in results:
            perf = results["performance_results"]
            print(f"\n⚡ PERFORMANCE RESULTS:")
            print(f"   Average Response Time: {perf.get('average_response_time', 0):.3f}s")
            print(f"   P95 Response Time: {perf.get('p95_response_time', 0):.3f}s")
            print(f"   Throughput: {perf.get('throughput', 0):.2f} requests/second")
            print(f"   Error Rate: {perf.get('error_rate', 0):.3f}")
            print(f"   Memory Usage: {perf.get('average_memory_usage', 0):.2f} MB")
            
            # Performance assessment
            avg_time = perf.get('average_response_time', 0)
            if avg_time < 3.0:
                print("   Performance Grade: A (Excellent) - Meets target < 3s")
            elif avg_time < 5.0:
                print("   Performance Grade: B (Good) - Meets target < 5s")
            else:
                print("   Performance Grade: C (Needs Improvement) - Above target")
        
        # Generate detailed report
        print(f"\n📋 GENERATING DETAILED REPORT...")
        report = evaluation_system.generate_evaluation_report(results)
        
        # Save report to file
        with open("evaluation_report.txt", "w") as f:
            f.write(report)
        
        print("✅ Evaluation completed successfully!")
        print("📄 Detailed report saved to 'evaluation_report.txt'")
        
        # Summary
        print(f"\n🎉 SUMMARY:")
        if "squad_results" in results:
            squad = results["squad_results"]
            f1 = squad.get('average_f1', 0)
            em = squad.get('exact_match_score', 0)
            
            if f1 >= 0.75 and em >= 0.65:
                print("   ✅ System meets production-quality SQUAD benchmarks!")
            elif f1 >= 0.7 or em >= 0.6:
                print("   ⚠️  System shows good performance but could be improved")
            else:
                print("   ❌ System needs significant improvement to meet benchmarks")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = asyncio.run(main())
