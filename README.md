# LocalLightEval

A comprehensive LLM evaluation benchmark using vLLM as the inference backend and HuggingFace Transformers. This tool provides efficient, scalable evaluation of large language models on binary classification tasks with rich configuration management and beautiful output formatting.

## Features

- **High-Performance Inference**: Built on vLLM for efficient GPU utilization and fast inference
- **Flexible Configuration**: Hydra-powered configuration system with modular, composable configs
- **Rich UI**: Beautiful console output with progress bars, tables, and formatted results
- **Comprehensive Metrics**: Detailed evaluation metrics including accuracy, precision, recall, F1, ROC-AUC
- **Output Management**: Organized output directories with timestamped runs and comprehensive logging
- **Error Analysis**: Detailed analysis of prediction errors and failure modes
- **GPU Support**: Automatic GPU detection and memory management
- **Extensible**: Modular design for easy extension to new tasks and models

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Conda (for environment management)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd locallighteval
```

2. **Create and activate conda environment**:
```bash
conda create -n locallighteval python=3.9
conda activate locallighteval
```

3. **Install the package**:
```bash
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Your input data should be a JSON Lines file where each line contains:
- `text`: The input text to classify
- `label`: Binary label (0 or 1, or boolean)

Example:
```json
{"text": "This movie is great!", "label": 1}
{"text": "Boring and poorly made.", "label": 0}
```

### 2. Basic Usage

Run evaluation with default settings:
```bash
python -m locallighteval.main data.input_path=/path/to/your/data.json
```

### 3. Using Different Models

Use any HuggingFace model:
```bash
python -m locallighteval.main data.input_path=/path/to/your/data.json model.name=meta-llama/Llama-2-7b-chat-hf
python -m locallighteval.main data.input_path=/path/to/your/data.json model.name=mistralai/Mistral-7B-Instruct-v0.1
```

Use a local model:
```bash
python -m locallighteval.main data.input_path=/path/to/your/data.json model.name=/path/to/local/model
```

### 4. Configuration Examples

With custom model parameters:
```bash
python -m locallighteval.main \
    data.input_path=/path/to/your/data.json \
    model.name=meta-llama/Llama-2-7b-chat-hf \
    model.tensor_parallel_size=2 \
    model.gpu_memory_utilization=0.8 \
    inference.batch_size=64 \
    inference.temperature=0.1
```

With few-shot prompting:
```bash
python -m locallighteval.main \
    data.input_path=/path/to/your/data.json \
    model.name=mistralai/Mistral-7B-Instruct-v0.1 \
    inference=few_shot
```

## Configuration System

LocalLightEval uses Hydra for powerful, flexible configuration management. Configurations are organized into modular components:

### Configuration Structure
```
config/
├── config.yaml              # Main config with model settings
├── inference/
│   ├── default.yaml         # Default inference params
│   └── few_shot.yaml        # Few-shot prompting
├── data/
│   └── default.yaml         # Data loading config
└── output/
    └── default.yaml         # Output management
```

### Model Configuration

Models are configured directly in the main config or via command line. The default config includes:

```yaml
model:
  name: "microsoft/DialoGPT-medium"  # Any HF repo or local path
  trust_remote_code: false
  tensor_parallel_size: 1
  dtype: "auto"
  max_model_len: null
  gpu_memory_utilization: 0.9
```

You can specify any model via command line:
```bash
# HuggingFace models
python -m locallighteval.main model.name=meta-llama/Llama-2-7b-chat-hf

# Local models  
python -m locallighteval.main model.name=/path/to/your/model

# With additional parameters
python -m locallighteval.main \
    model.name=mistralai/Mistral-7B-Instruct-v0.1 \
    model.trust_remote_code=true \
    model.tensor_parallel_size=2
```

Create custom inference settings:
```yaml
# config/inference/my_inference.yaml
# @package inference
max_tokens: 50
temperature: 0.3
batch_size: 32
instruction: "Your custom classification instruction here."
```

## Output Structure

Each evaluation run creates a timestamped directory with comprehensive outputs:

```
outputs/
└── eval_20240827_143022/
    ├── config.yaml              # Complete configuration used
    ├── evaluation.log           # Detailed execution logs
    ├── metrics.json             # Evaluation metrics
    ├── detailed_results.json    # Per-sample results
    ├── detailed_results.csv     # CSV format for analysis
    ├── predictions.json         # Raw model predictions
    ├── error_analysis.json      # Error pattern analysis
    ├── run_metadata.json        # Run metadata and system info
    └── run_summary.txt          # Human-readable summary
```

## Available Models

The system works with **any vLLM-compatible model**, including:

### HuggingFace Models
- **Llama 2**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Code Llama**: `codellama/CodeLlama-7b-Instruct-hf`
- **Vicuna**: `lmsys/vicuna-7b-v1.5`, `lmsys/vicuna-13b-v1.5`
- **And thousands more**: Any model on [HuggingFace Hub](https://huggingface.co/models)

### Local Models
- Downloaded HF models: `/path/to/downloaded/model`
- Custom fine-tuned models: `/path/to/your/custom/model`
- Converted models in HF format

### Usage Examples
```bash
# Popular open models
python -m locallighteval.main model.name=meta-llama/Llama-2-7b-chat-hf
python -m locallighteval.main model.name=mistralai/Mistral-7B-Instruct-v0.1
python -m locallighteval.main model.name=microsoft/DialoGPT-large

# Local models
python -m locallighteval.main model.name=/home/user/models/my-fine-tuned-llama
```

## Performance Optimization

### GPU Configuration
- Adjust `model.tensor_parallel_size` for multi-GPU inference
- Tune `model.gpu_memory_utilization` based on your GPU memory
- Use `model.dtype="bfloat16"` for memory efficiency

### Batch Processing
- Increase `inference.batch_size` for better throughput
- Balance batch size with available GPU memory
- Monitor GPU utilization during evaluation

### Memory Management
- Set `model.max_model_len` to limit context length
- Use smaller models for development and testing
- Monitor system resources during evaluation

## Metrics and Analysis

LocalLightEval provides comprehensive evaluation metrics:

### Core Classification Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives) 
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)
- **ROC-AUC**: Area under the ROC curve (when probabilities available)

### Additional Analysis
- **Confusion Matrix**: Detailed breakdown of predictions
- **Error Analysis**: Common failure patterns and examples
- **Response Parsing**: Statistics on model response interpretation
- **Per-Sample Results**: Individual predictions with metadata

## Development

### Project Structure
```
locallighteval/
├── locallighteval/
│   ├── __init__.py
│   ├── main.py              # Main evaluation script
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # Data loading utilities
│   ├── inference.py         # vLLM inference engine
│   ├── metrics.py           # Metrics computation
│   └── utils.py             # Utilities and Rich formatting
├── config/                  # Configuration files
├── pyproject.toml           # Package configuration
└── example_run.py           # Example usage script
```

### Running Examples

Generate sample data and test the system:
```bash
python example_run.py
```

This creates sample data and shows example commands for evaluation.

### Environment Validation

The system automatically validates that all required dependencies are installed:
- vLLM for inference
- Transformers for model support
- Rich for beautiful output
- Hydra for configuration
- scikit-learn for metrics
- And more...

## Troubleshooting

### Common Issues

**GPU Memory Errors**:
- Reduce `inference.batch_size`
- Lower `model.gpu_memory_utilization`
- Use smaller model or reduce `model.max_model_len`

**Model Loading Issues**:
- Verify model name/path is correct
- Check if model requires `trust_remote_code: true`
- Ensure sufficient disk space for model downloads

**Configuration Errors**:
- Validate YAML syntax in config files
- Check that required fields are provided
- Verify file paths exist

**Performance Issues**:
- Monitor GPU utilization
- Adjust batch size and tensor parallelism
- Consider model quantization for memory efficiency

### Getting Help

- Check the evaluation logs in the output directory
- Review error analysis for prediction issues
- Examine the configuration summary for parameter verification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- Uses [Hydra](https://github.com/facebookresearch/hydra) for configuration management
- Powered by [Rich](https://github.com/Textualize/rich) for beautiful console output
- Leverages [HuggingFace Transformers](https://github.com/huggingface/transformers) ecosystem 
