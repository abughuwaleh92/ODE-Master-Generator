# setup.py - Enhanced Setup Configuration
# ============================================================================
"""
Setup configuration for Master Generators ODE System
"""

from setuptools import setup, find_packages
import os
import codecs

def read_file(filename):
    """Read file content"""
    with codecs.open(filename, encoding='utf-8') as f:
        return f.read()

def get_version():
    """Get version from __init__.py"""
    with open('src/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '2.0.0'

# Read requirements
def get_requirements(filename='requirements.txt'):
    """Parse requirements file"""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Long description from README
long_description = read_file('README.md') if os.path.exists('README.md') else ''

setup(
    name='master-generators-ode',
    version=get_version(),
    author='Master Generators Team',
    author_email='contact@master-generators.com',
    description='Master Generators for ODEs using ML/DL - Enhanced Implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/master-generators/master-generators-ode',
    project_urls={
        'Bug Tracker': 'https://github.com/master-generators/master-generators-ode/issues',
        'Documentation': 'https://master-generators.readthedocs.io',
        'Source Code': 'https://github.com/master-generators/master-generators-ode',
    },
    
    packages=find_packages(exclude=['tests*', 'docs*', 'scripts*', 'notebooks*']),
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Framework :: FastAPI',
        'Framework :: Streamlit',
    ],
    
    python_requires='>=3.8',
    
    install_requires=get_requirements(),
    
    extras_require={
        'dev': get_requirements('requirements-dev.txt') if os.path.exists('requirements-dev.txt') else [],
        'docs': ['sphinx>=7.0.0', 'sphinx-rtd-theme>=1.3.0'],
        'test': ['pytest>=7.4.0', 'pytest-cov>=4.1.0', 'pytest-asyncio>=0.21.0'],
    },
    
    entry_points={
        'console_scripts': [
            'master-generators=src.cli:main',
            'mg-api=api_server:main',
            'mg-app=master_generators_app:main',
            'mg-train=src.ml.trainer:train_cli',
            'mg-analyze=src.dl.novelty_detector:analyze_cli',
        ],
    },
    
    package_data={
        'src': ['*.json', '*.yaml', '*.yml'],
        'src.data': ['*.csv', '*.json'],
        'src.models': ['*.pth', '*.h5', '*.pkl'],
        'src.static': ['*.css', '*.js', '*.html'],
    },
    
    include_package_data=True,
    zip_safe=False,
    
    keywords=[
        'ODE', 'differential equations', 'machine learning', 'deep learning',
        'mathematics', 'scientific computing', 'neural networks', 'generators',
        'special functions', 'numerical analysis', 'symbolic computation',
        'research', 'education'
    ],
)
