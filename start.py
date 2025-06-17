#!/usr/bin/env python3
"""
Simple startup script for the Enhanced Q&A System
"""
import subprocess
import sys
import time
import socket
import os

def check_port(port):
    """Check if port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except:
        return False

def main():
    print("🤖 Enhanced Document Q&A System v2.0")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app"):
        print("❌ Please run this script from the project root directory")
        print("   Make sure you can see the 'app' folder")
        return
    
    # Check backend port
    if not check_port(8000):
        print("⚠️  Port 8000 is in use. Backend might already be running.")
        print("   Visit http://127.0.0.1:8000 to check")
    
    # Find available port for frontend
    frontend_port = 8501
    if not check_port(frontend_port):
        frontend_port = 8502
    
    print(f"🚀 Starting backend on port 8000...")
    backend = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ])
    
    print("⏳ Waiting for backend to start...")
    time.sleep(5)
    
    print(f"🎨 Starting frontend on port {frontend_port}...")
    frontend = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
        "--server.port", str(frontend_port),
        "--server.address", "127.0.0.1"
    ])
    
    print(f"\n✅ System started successfully!")
    print(f"📍 Backend: http://127.0.0.1:8000")
    print(f"📍 Frontend: http://127.0.0.1:{frontend_port}")
    print(f"📍 API Docs: http://127.0.0.1:8000/docs")
    print(f"\n🌐 Open your browser: http://127.0.0.1:{frontend_port}")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        backend.terminate()
        frontend.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main()
