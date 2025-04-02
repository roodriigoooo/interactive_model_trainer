"""
Test script to ensure proper installation of dependencies.
"""

def test_dependencies():
    """Test the installation of required dependencies."""
    print("Testing dependencies installation...")
    
    try:
        import streamlit
        print("✓ Streamlit installed")
    except ImportError:
        print("✗ Streamlit not found. Install with: pip install streamlit")
    
    try:
        import pandas
        print("✓ Pandas installed")
    except ImportError:
        print("✗ Pandas not found. Install with: pip install pandas")
    
    try:
        import numpy
        print("✓ NumPy installed")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
    
    try:
        import sklearn
        print("✓ Scikit-learn installed")
    except ImportError:
        print("✗ Scikit-learn not found. Install with: pip install scikit-learn")
    
    try:
        import seaborn
        print("✓ Seaborn installed")
    except ImportError:
        print("✗ Seaborn not found. Install with: pip install seaborn")
    
    try:
        import matplotlib
        print("✓ Matplotlib installed")
    except ImportError:
        print("✗ Matplotlib not found. Install with: pip install matplotlib")
    
    try:
        import joblib
        print("✓ Joblib installed")
    except ImportError:
        print("✗ Joblib not found. Install with: pip install joblib")
    
    print("\nDone testing dependencies.")

def test_app():
    """Test if the app is properly setup."""
    print("Testing app setup...")
    
    import os
    
    # Check if required files exist
    if os.path.exists("app.py"):
        print("✓ Main app.py file found")
    else:
        print("✗ Main app.py file not found")
    
    if os.path.exists("requirements.txt"):
        print("✓ Requirements file found")
    else:
        print("✗ Requirements file not found")
    
    # Check if modules are properly set up
    if os.path.exists("utils") and os.path.exists("utils/__init__.py"):
        print("✓ Utils package found")
    else:
        print("✗ Utils package not properly set up")
    
    if os.path.exists("models") and os.path.exists("models/__init__.py"):
        print("✓ Models package found")
    else:
        print("✗ Models package not properly set up")
    
    print("\nDone testing app setup.")

if __name__ == "__main__":
    print("=" * 50)
    print("ML Model Trainer Installation Test")
    print("=" * 50)
    
    test_dependencies()
    print("\n" + "-" * 50 + "\n")
    test_app()
    
    print("\n" + "=" * 50)
    print("If all tests passed, you can run the app with: streamlit run app.py")
    print("=" * 50) 