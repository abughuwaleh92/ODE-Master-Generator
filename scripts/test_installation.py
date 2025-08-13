"""
test_installation.py - Test if all packages are installed correctly
"""

import sys
import importlib

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def test_version(module_name):
    """Get module version if available"""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, '__version__', 'unknown')
    except Exception:
        return None

def run_tests():
    """Run all installation tests"""
    print("Testing Master Generators Installation")
    print("=" * 50)

    packages = {
        'streamlit': '1.28.0',
        'numpy': '1.24.0',
        'sympy': '1.12',
        'torch': '2.0.0',
        'matplotlib': '3.7.0',
        'scipy': '1.11.0',
        'pandas': '2.0.0',
        'transformers': '4.35.0',
        'tensorflow': '2.14.0',
        'sklearn': None,
        'plotly': '5.18.0',
        'fastapi': '0.104.0',
        'uvicorn': '0.24.0',
        'pydantic': '2.5.0'
    }

    results = []
    all_passed = True

    for package, min_version in packages.items():
        success, error = test_import(package)
        version = test_version(package) if success else None

        if success:
            status = "✅ PASS"
            print(f"{status} {package:20} Version: {version}")
        else:
            status = "❌ FAIL"
            print(f"{status} {package:20} Error: {error}")
            all_passed = False

        results.append({
            'package': package,
            'success': success,
            'version': version,
            'required': min_version,
            'error': error
        })

    print("=" * 50)

    # GPU checks (best-effort)
    print("\nGPU Check:")
    print("-" * 30)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  No CUDA GPU available, using CPU")
    except Exception:
        print("⚠️  Could not check PyTorch CUDA availability")

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: {len(gpus)} device(s) found")
        else:
            print("⚠️  TensorFlow: No GPU found")
    except Exception:
        pass

    print("=" * 50)

    if all_passed:
        print("✅ All packages installed successfully!")
        return 0
    else:
        print("❌ Some packages are missing or failed to import")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

def check_system_info():
    """Display system information"""
    import platform
    print("\nSystem Information:")
    print("-" * 30)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")

    try:
        import psutil
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"CPU Cores: {psutil.cpu_count()}")
    except Exception:
        pass

def main():
    """Main function"""
    check_system_info()
    print()
    return run_tests()

if __name__ == "__main__":
    sys.exit(main())
