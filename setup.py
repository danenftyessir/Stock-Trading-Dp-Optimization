"""
Setup script for Stock Trading Optimization with Dynamic Programming.

Installation:
    pip install -e .  # Development installation
    pip install .     # Regular installation

Usage after installation:
    from stock_trading_optimization import DynamicProgrammingTrader
    trader = DynamicProgrammingTrader(max_transactions=5)
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Dynamic Programming Stock Trading Optimization"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Package metadata
PACKAGE_NAME = "stock-trading-dp-optimization"
VERSION = "1.0.0"
DESCRIPTION = "Dynamic Programming approach for optimizing stock trading profits with k-transaction constraints"
AUTHOR = "Danendra Shafi Athallah"
AUTHOR_EMAIL = "danendra1967@gmail.com"
URL = "https://github.com/danendra/stock-trading-dp-optimization"
LICENSE = "MIT"

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Office/Business :: Financial :: Investment",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]

# Keywords
KEYWORDS = [
    "dynamic programming",
    "stock trading",
    "algorithmic trading",
    "financial optimization",
    "k-transactions",
    "backtesting",
    "alpha vantage",
    "quantitative finance",
    "portfolio optimization",
    "trading strategies"
]

# Package data
PACKAGE_DATA = {
    "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
}

# Entry points for command-line scripts
ENTRY_POINTS = {
    "console_scripts": [
        "stock-trading-collect=scripts.data_collection:main",
        "stock-trading-backtest=scripts.run_backtesting:main",
        "stock-trading-optimize=scripts.parameter_optimization:main",
        "stock-trading-report=scripts.generate_reports:main",
        "stock-trading-pipeline=scripts.full_pipeline:main",
    ],
}

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "pre-commit>=2.20.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=0.18.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "ipywidgets>=8.0.0",
    ],
    "cloud": [
        "boto3>=1.26.0",  # AWS
        "google-cloud-storage>=2.7.0",  # Google Cloud
        "azure-storage-blob>=12.14.0",  # Azure
    ],
    "monitoring": [
        "sentry-sdk>=1.15.0",
        "prometheus-client>=0.15.0",
    ],
    "all": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "sphinx>=5.0.0",
        "jupyter>=1.0.0",
        "boto3>=1.26.0",
        "sentry-sdk>=1.15.0",
    ],
}

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # Package discovery
    packages=find_packages(where=".", exclude=["tests*", "docs*", "scripts*"]),
    package_dir={"": "."},
    
    # Dependencies
    install_requires=read_requirements(),
    extras_require=EXTRAS_REQUIRE,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package metadata
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    
    # Package data
    package_data=PACKAGE_DATA,
    include_package_data=True,
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/danendra/stock-trading-dp-optimization/docs",
        "Source": "https://github.com/danendra/stock-trading-dp-optimization",
        "Tracker": "https://github.com/danendra/stock-trading-dp-optimization/issues",
        "Research Paper": "https://example.com/paper",  # Update with actual paper URL
    },
    
    # Additional metadata
    zip_safe=False,
    test_suite="tests",
)

# Post-installation message
print("""
🎉 Stock Trading Optimization with Dynamic Programming installed successfully!

📋 Quick Start:
    1. Copy configuration: cp .env.example .env
    2. Set your API key in .env: ALPHA_VANTAGE_API_KEY=JHU900LRGUA6T7F7
    3. Run data collection: stock-trading-collect --symbols AAPL GOOGL
    4. Run backtesting: stock-trading-backtest --strategy dp --k-values 2 5 10
    5. Generate reports: stock-trading-report --latest

📖 Documentation: Check README.md and docs/ directory
🧪 Run tests: pytest tests/
💡 Example notebooks: See examples/ directory

For support and questions:
📧 Contact: danendra1967@gmail.com
🏫 Institution: Institut Teknologi Bandung
""")