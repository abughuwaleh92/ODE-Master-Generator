# Master Generators for ODEs - ML/DL Implementation

## Overview

This project implements the research paper "Master Generators: A Novel Approach to Construct and Solve Ordinary Differential Equations" using advanced Machine Learning and Deep Learning techniques.

## Features

### 1. **Master Generators Implementation**
- Complete implementation of Theorems 4.1 and 4.2 from the research paper
- Support for both linear and non-linear generators (Tables 1 & 2)
- Infinite generator creation using various f(z) functions

### 2. **Function Support**
- **Basic Functions**: sin(z), cos(z), tanh(z), ln(z), e^(z), and more
- **Special Functions**: Airy functions (Ai, Bi), Gamma function, Bessel functions, Legendre polynomials, Hermite polynomials, Chebyshev polynomials, Zeta function

### 3. **Machine Learning Module**
- Pattern learning from generated ODEs
- Neural network architecture for understanding generator patterns
- Automatic generation of new ODEs based on learned patterns
- Training on customizable datasets

### 4. **Deep Learning Novelty Detection**
- Transformer-based architecture for ODE analysis
- Detection of novel and unsolvable ODEs
- Complexity assessment
- Recommendation of appropriate solution methods

### 5. **Full API & UI**
- **Streamlit UI**: Interactive web interface
- **FastAPI Backend**: RESTful API for programmatic access
- **Batch Generation**: Generate multiple ODEs simultaneously
- **Real-time Analysis**: Instant novelty and complexity assessment

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/master-generators.git
cd master-generators
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run master_generators_app.py
```

4. Run the API server (in another terminal):
```bash
python api_server.py
```

### Docker Installation

```bash
docker build -t master-generators .
docker run -p 8501:8501 -p 8000:8000 master-generators
```

## Railway Deployment

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.json` configuration
3. Deploy with one click

## Usage

### Web Interface

Access the Streamlit interface at `http://localhost:8501`

#### Single Generator Mode
1. Select generator type (Linear/Nonlinear)
2. Choose a function f(z)
3. Set parameters (α, β, n, M)
4. Click "Generate ODE"

#### Batch Generation
1. Set number of ODEs to generate
2. Select types to include
3. Choose functions
4. Click "Generate Batch"

#### ML Pattern Learning
1. Set training parameters
2. Click "Train Model"
3. Generate new ODEs using ML

#### Novelty Detection
1. Generate or input an ODE
2. Click "Analyze"
3. View novelty score and recommendations

### API Usage

The API is available at `http://localhost:8000`

#### Generate Single ODE
```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate/single",
    json={
        "type": "nonlinear",
        "generator_number": 2,
        "function": "bessel_j0",
        "parameters": {
            "alpha": 1.0,
            "beta": 2.0,
            "n": 3,
            "M": 0.0
        },
        "q": 3,
        "v": 2
    }
)

result = response.json()
print(result)
```

#### Batch Generation
```python
response = requests.post(
    "http://localhost:8000/api/generate/batch",
    json={
        "count": 10,
        "types": ["linear", "nonlinear"],
        "functions": ["sine", "cosine", "gamma"],
        "random_params": True
    }
)
```

#### ML Generation
```python
# Train the model
requests.post(
    "http://localhost:8000/api/ml/train",
    json={
        "epochs": 100,
        "batch_size": 32,
        "samples": 1000
    }
)

# Generate using ML
response = requests.get("http://localhost:8000/api/ml/generate")
```

## Mathematical Background

### Theorem 4.1
The implementation follows the fundamental theorem from the paper:

For an analytic function f in disc D centered at α ∈ ℝ:
- y(x) = π/(2n) Σ[s=1 to n] (2f(α+β) - (ψ(α,ω,x) + φ(α,ω,x))) + πM

Where:
- ω = (2s-1)π/(2n)
- ψ(α,ω,x) = f(α + β*e^(ix*cos(ω) - x*sin(ω)))
- φ(α,ω,x) = f(α + β*e^(-ix*cos(ω) - x*sin(ω)))

### Linear Generators
Implements generators from Table 1, including:
1. y''(x) + y(x) = RHS
2. y''(x) + y'(x) = RHS
3. y(x) + y'(x) = RHS
4. Pantograph equations

### Nonlinear Generators
Implements generators from Table 2, including:
1. (y''(x))^q + y(x) = RHS
2. (y''(x))^q + (y'(x))^v = RHS
3. y(x) + (y'(x))^v = RHS
4. Exponential and trigonometric nonlinearities

## Architecture

### Core Components

1. **MasterGenerator**: Base class implementing theorems
2. **GeneratorFactory**: Creates various generator types
3. **ODEGeneratorML**: Machine learning pattern learner
4. **ODENoveltyDetector**: Deep learning novelty detector
5. **ODEGeneratorApp**: Streamlit UI application
6. **FastAPI Server**: RESTful API backend

### ML Architecture

- **Encoder-Decoder Network**: For pattern learning
- **Transformer Model**: For novelty detection
- **Feature Extraction**: Parameters encoding
- **Pattern Recognition**: Generator type identification

## Applications

The system can be applied to:
- Quantum field theory
- Astrophysics
- Fluid dynamics
- Economics modeling
- Engineering systems
- Mathematical research

## Performance

- Generates 100+ ODEs in seconds
- ML training: ~100 epochs in minutes
- Real-time novelty detection
- Handles special functions efficiently

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Citation

If you use this implementation, please cite:

```bibtex
@article{abu2024master,
  title={Master Generators: A Novel Approach to Construct and Solve Ordinary Differential Equations},
  author={Abu-Ghuwaleh, Mohammad and Saadeh, Rania and Saffaf, Rasheed},
  journal={Journal of Applied Mathematics},
  year={2024},
  publisher={Zarqa University}
}
```

## Authors

- Implementation based on research by Mohammad Abu-Ghuwaleh.et.al
- ML/DL implementation developed for practical applications

## Support

For issues or questions:
- Open an issue on GitHub
- Contact via email: [20209030@zu.edu.jo]

## Acknowledgments

Special thanks to the original research authors for their groundbreaking work on Master Generators.