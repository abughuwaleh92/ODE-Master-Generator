"""
Minimal startup check for Railway deployment
"""

import os
import sys

def main():
    print("Master Generators Startup Check")
    print("=" * 50)
    
    # Create necessary directories
    directories = ['models', 'data', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Check critical imports
    try:
        import streamlit
        import numpy
        import sympy
        import pandas
        print("✅ All critical packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        sys.exit(1)
    
    print("✅ System ready!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
