#!/usr/bin/env python
"""
BankGPT Startup Script
Automatically starts FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    """Print startup header"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}{Colors.BLUE}BankGPT FastAPI + Streamlit Startup{Colors.RESET}")
    print("="*60 + "\n")

def check_env_file():
    """Check if .env file exists"""
    if os.path.exists(".env"):
        print(f"{Colors.GREEN}✓{Colors.RESET} .env file configured")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.RESET} .env file not found")
        print(f"  Please create .env from .env.example and add your API keys")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "google-generativeai",
        "groq",
        "httpx",
        "streamlit",
        "chromadb"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"{Colors.RED}✗{Colors.RESET} Missing packages: {', '.join(missing_packages)}")
        print(f"{Colors.YELLOW}⚠{Colors.RESET} Installing dependencies...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + missing_packages,
                check=True,
                timeout=300
            )
            print(f"{Colors.GREEN}✓{Colors.RESET} Dependencies installed")
            return True
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}✗{Colors.RESET} Installation timeout")
            return False
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗{Colors.RESET} Installation failed: {e}")
            return False
    else:
        print(f"{Colors.GREEN}✓{Colors.RESET} All dependencies installed")
        return True

def start_backend():
    """Start FastAPI backend"""
    print(f"\n{Colors.BLUE}Starting FastAPI Backend...{Colors.RESET}")
    
    try:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--reload"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for backend to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"{Colors.GREEN}✓{Colors.RESET} Backend running on http://127.0.0.1:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"{Colors.RED}✗{Colors.RESET} Backend failed to start")
            print(f"  Error: {stderr}")
            return None
            
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.RESET} Failed to start backend: {e}")
        return None

def start_frontend():
    """Start Streamlit frontend"""
    print(f"\n{Colors.BLUE}Starting Streamlit Frontend...{Colors.RESET}")
    
    try:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                "app.py",
                "--server.port", "8501",
                "--server.address", "127.0.0.1"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for frontend to start
        time.sleep(5)
        
        if process.poll() is None:
            print(f"{Colors.GREEN}✓{Colors.RESET} Frontend running on http://127.0.0.1:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"{Colors.RED}✗{Colors.RESET} Frontend failed to start")
            print(f"  Error: {stderr}")
            return None
            
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.RESET} Failed to start frontend: {e}")
        return None

def monitor_processes(backend, frontend):
    """Monitor running processes"""
    print(f"\n{Colors.GREEN}{'='*60}")
    print(f"Both services are running!")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n{Colors.BOLD}Access the application:{Colors.RESET}")
    print(f"  Frontend: {Colors.BLUE}http://127.0.0.1:8501{Colors.RESET}")
    print(f"  Backend:  {Colors.BLUE}http://127.0.0.1:8000{Colors.RESET}")
    print(f"  API Docs: {Colors.BLUE}http://127.0.0.1:8000/docs{Colors.RESET}")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop both services{Colors.RESET}\n")
    
    try:
        while True:
            if backend and backend.poll() is not None:
                print(f"{Colors.RED}✗{Colors.RESET} Backend process died")
                break
            if frontend and frontend.poll() is not None:
                print(f"{Colors.RED}✗{Colors.RESET} Frontend process died")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        
        # Terminate processes
        if backend:
            backend.terminate()
            try:
                backend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.kill()
        
        if frontend:
            frontend.terminate()
            try:
                frontend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend.kill()
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Services stopped gracefully")
        sys.exit(0)

def main():
    """Main startup function"""
    print_header()
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start services
    backend = start_backend()
    if not backend:
        sys.exit(1)
    
    frontend = start_frontend()
    if not frontend:
        if backend:
            backend.terminate()
        sys.exit(1)
    
    # Monitor processes
    monitor_processes(backend, frontend)

if __name__ == "__main__":
    main()
