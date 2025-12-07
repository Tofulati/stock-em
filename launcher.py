#!/usr/bin/env python3
"""
Simple Launcher for Trading Model Dashboard
Starts the backend server and opens the dashboard
"""

import subprocess
import webbrowser
import time
import sys
import os
from pathlib import Path

def main():
    print("\n" + "=" * 60)
    print("Trading Model Dashboard Launcher")
    print("=" * 60 + "\n")
    
    # Check if backend.py exists
    if not Path('backend.py').exists():
        print("Error: backend.py not found")
        print("   Make sure you're in the correct directory")
        sys.exit(1)
    
    # Check if dashboard.html exists
    if not Path('dashboard.html').exists():
        print("Error: dashboard.html not found")
        print("   Make sure you're in the correct directory")
        sys.exit(1)
    
    print("Files found")
    print("\nStarting backend server...")
    
    # Start backend server
    try:
        backend_process = subprocess.Popen(
            [sys.executable, 'backend.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except Exception as e:
        print(f"Failed to start backend: {e}")
        sys.exit(1)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Check if server is running
    if backend_process.poll() is not None:
        print("Backend server failed to start")
        stdout, _ = backend_process.communicate()
        print(stdout)
        sys.exit(1)
    
    print("Backend server started on http://localhost:5000")
    
    # Open dashboard
    print("\nOpening dashboard in browser...")
    dashboard_path = Path('dashboard.html').absolute()
    
    try:
        webbrowser.open(f'file://{dashboard_path}')
        print(f"Dashboard opened: {dashboard_path}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"   Please open manually: {dashboard_path}")
    
    # Success message
    print("\n" + "=" * 60)
    print("Dashboard is ready!")
    print("=" * 60)
    print("\n Tips:")
    print("   • Enter a ticker and click 'Run Backtest'")
    print("   • Or upload an existing JSON file")
    print("   • Or use 'Watch Mode' for auto-loading")
    print("\n Backend API: http://localhost:5000")
    print(" Dashboard: file://" + str(dashboard_path))
    print("\n Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        # Stream backend logs
        print(" Backend Server Logs:")
        print("-" * 60)
        
        for line in iter(backend_process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                
    except KeyboardInterrupt:
        print("\n\n Shutting down...")
        backend_process.terminate()
        backend_process.wait(timeout=5)
        print(" Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n  Error: {e}")
        backend_process.terminate()

if __name__ == '__main__':
    main()