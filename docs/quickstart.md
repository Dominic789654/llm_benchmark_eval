# Quick Start Guide

This guide will help you get started with the LLM Benchmark Evaluation Framework.

## Prerequisites

1. **Python Environment**: Python 3.8+ with required packages
2. **GPU Access**: NVIDIA GPU with sufficient memory for model inference
3. **Model Access**: Access to the language models you want to evaluate
4. **Dataset Setup**: Benchmark datasets in the expected format

## Installation

```bash
# Clone or extract the framework
cd llm_benchmark_eval

# Install dependencies (create requirements.txt first)
pip install torch transformers datasets vllm pyyaml tqdm loguru
```

## Basic Usage

### 1. Prepare Your Data

Ensure your benchmark datasets are in the `data/` directory with the following structure:

```
data/
├── math/
│   └── test.jsonl
├── gsm8k/
│   └── test.jsonl
├── aime/
│   └── test2024.jsonl
└── ...
```

Each JSONL file should contain entries like:
```json
{"problem": "What is 2+2?", "answer": "4"}
```

### 2. Configure Your Evaluation

Copy and modify a configuration file:

```bash
cp configs/basic_eval.yaml my_config.yaml
```

Edit `my_config.yaml` to specify:
- Your model path
- Which datasets to evaluate
- Output directory
- Batch sizes and other parameters

### 3. Run Evaluation

```bash
python scripts/run_benchmark.py --config my_config.yaml
```

### 4. View Results

Results will be saved in the specified output directory:
- `{dataset}_results.jsonl`: Individual results for each sample
- `{dataset}_metrics.json`: Computed metrics for the dataset
- `evaluation_summary.json`: Overall summary across all datasets

## Advanced Usage

### Comparing Multiple Models

```bash
python scripts/compare_models.py \
    --base_config configs/model_comparison.yaml \
    --models /path/to/model1 /path/to/model2 \
    --model_names "Model A" "Model B" \
    --output_dir ./comparison_results
```

### Custom Evaluation

For more control, you can use the framework programmatically:

```python
from src.datasets import MathDatasetLoader
from src.models import VLLMModel
from src.evaluators import XVerifyEvaluator

# Load dataset
loader = MathDatasetLoader("./data")
samples = loader.load_split("test", max_samples=10)

# Initialize model
model = VLLMModel("/path/to/your/model")

# Initialize evaluator
evaluator = XVerifyEvaluator()

# Run evaluation
# ... (see examples/ for complete code)
```

## Configuration Options

### Model Configuration

```yaml
model:
  path: "/path/to/model"           # Model path or HF model ID
  backend: "vllm"                  # "vllm" or "huggingface"
  use_vllm: true                   # Enable vLLM acceleration
  temperature: 0.7                 # Generation temperature
  max_tokens: 2048                 # Maximum tokens to generate
  system_prompt: "..."             # Optional system prompt
```

### Dataset Configuration

```yaml
datasets:
  data_dir: "./data"               # Root data directory
  datasets: ["math", "gsm8k"]     # List of datasets to evaluate
  split: "test"                    # Dataset split to use
  max_samples: 100                 # Limit number of samples
```

### Evaluator Configuration

```yaml
evaluator:
  type: "xverify"                  # Evaluator type
  model_path: "/path/to/xverify"   # xVerify model path
  use_vllm: true                   # Use vLLM for evaluation
  process_num: 5                   # Number of parallel processes
```

## Supported Datasets

| Dataset | Description | Samples | Domain |
|---------|-------------|---------|--------|
| MATH | Competition mathematics | 5,000 | High school math |
| GSM8K | Grade school math | 1,319 | Elementary math |
| AIME | Math competition | 360 | Advanced math |
| AIME25 | Recent AIME problems | 30 | Advanced math |
| OlympiadBench | Math olympiad | 8,476 | Expert math |
| Minerva | STEM reasoning | 272 | Mixed STEM |
| GPQA | Graduate physics | 448 | Graduate physics |
| MMLU-Pro | Multi-domain | 12,000 | General knowledge |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `batch_size` in configuration
- Lower `gpu_memory_utilization` for vLLM
- Use smaller `max_tokens`

**2. Model Loading Fails**
- Check model path is correct
- Ensure sufficient disk space
- Verify model format compatibility

**3. Dataset Not Found**
- Check `data_dir` path in configuration
- Ensure dataset files exist with correct names
- Verify JSONL format is correct

**4. xVerify Evaluation Errors**
- Check xVerify model path
- Ensure xVerify dependencies are installed
- Try disabling vLLM for evaluation (`use_vllm: false`)

### Performance Tips

1. **Use vLLM**: Significantly faster than HuggingFace for inference
2. **Batch Processing**: Increase batch sizes if memory allows
3. **Parallel Evaluation**: Use multiple processes for xVerify evaluation
4. **Dataset Sampling**: Use `max_samples` for quick testing

## Getting Help

1. Check the configuration examples in `configs/`
2. Look at code examples in `examples/`
3. Review the API documentation in `docs/`
4. Check logs for detailed error messages
