# LLM Benchmark Evaluation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, standalone framework for evaluating Large Language Models on mathematical reasoning benchmarks using xVerify for automated assessment.

> **Note**: This framework requires appropriate dataset files and model access. Please refer to [`./data/README.md`](./data/README.md) for dataset preparation instructions.

## 🎯 Features

- **Multi-Dataset Support**: MATH, GSM8K, AIME, AIME25, AMC, OlympiadBench, Minerva, GPQA, MMLU-Pro
- **Universal LLM Interface**: Support for any LLM with vLLM acceleration
- **xVerify Integration**: Automated evaluation with vLLM-accelerated xVerify
- **Flexible Configuration**: YAML-based configuration system
- **Comprehensive Metrics**: Pass@k, token efficiency, strategy comparison
- **Parallel Processing**: Multi-process evaluation for efficiency
- **Easy Setup**: One-command installation and setup

## 📁 Structure

```
llm_benchmark_eval/
├── src/
│   ├── datasets/          # Dataset loaders and utilities
│   ├── models/            # LLM interfaces and wrappers
│   ├── evaluators/        # Evaluation engines
│   └── utils/             # Common utilities
├── configs/               # Configuration files
├── scripts/               # Execution scripts
├── examples/              # Usage examples
├── docs/                  # Documentation
└── data/                  # Benchmark datasets (to be populated)
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and setup the framework
git clone <repository>
cd llm_benchmark_eval

# Run setup script
./setup.sh

# Or manual setup:
pip install -r requirements.txt
mkdir -p data results logs
```

### 2. Setup Datasets

The framework supports the following datasets: **MATH**, **GSM8K**, **AIME**, **AIME25**, **AMC**, **OlympiadBench**, **Minerva**, **GPQA**, **MMLU-Pro**.

Prepare your datasets in the `data/` directory. See [`./data/README.md`](./data/README.md) for detailed instructions on dataset format and setup.

```bash
# Validate your datasets
python scripts/validate_datasets.py --data_dir ./data --verbose
```

### 3. Configure

Edit `my_config.yaml` (created by setup) with your model paths:

```yaml
model:
  path: "/path/to/your/model"
  backend: "vllm"  # or "huggingface"
  
datasets:
  data_dir: "./data"
  datasets: ["math", "gsm8k"]
  max_samples: 100
```

### 4. Run Evaluation

```bash
# Basic evaluation
python scripts/run_benchmark.py --config my_config.yaml

# Multi-model comparison
python scripts/compare_models.py \
    --base_config configs/model_comparison.yaml \
    --models /path/to/model1 /path/to/model2 \
    --model_names "Model A" "Model B"

# Programmatic usage
python examples/basic_evaluation.py
```

## 📊 Supported Benchmarks

The framework supports the following mathematical reasoning benchmarks:
- **MATH**, **GSM8K**, **AIME**, **AIME25**, **AMC** - Competition mathematics
- **OlympiadBench**, **Minerva** - Advanced reasoning
- **GPQA**, **MMLU-Pro** - Multi-domain evaluation

For dataset details and setup, see [`./data/README.md`](./data/README.md).

## 🔧 Configuration

All evaluation parameters are controlled via YAML configuration files, supporting:
- Model selection and parameters
- Dataset filtering and sampling
- Evaluation metrics and thresholds
- Output formatting and storage
- Parallel processing settings

## 📈 Metrics

- **Accuracy**: Standard correctness measurement
- **Pass@k**: Success rate within k attempts
- **Token Efficiency**: Tokens per problem/correct answer
- **Strategy Analysis**: Performance by generation strategy
- **Error Analysis**: Detailed failure categorization

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Adding new datasets
- Submitting pull requests

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_benchmark_eval,
  title = {LLM Benchmark Evaluation Framework},
  author = {Contributors},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/llm_benchmark_eval}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [xVerify](https://huggingface.co/IAAR-Shanghai/xVerify-0.5B-I) for automated evaluation
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- All the benchmark dataset creators and maintainers

## 📧 Contact

For questions and discussions:
- Open an issue on GitHub
- Check existing issues and discussions

## ⚠️ Disclaimer

This framework is provided as-is for research purposes. Ensure you have appropriate licenses and permissions for:
- The models you evaluate
- The datasets you use
- The evaluation models (e.g., xVerify)
