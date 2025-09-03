# LocalLightEval

A comprehensive LLM evaluation and clinical text summarization framework using vLLM as the inference backend. This tool provides efficient, scalable evaluation of large language models on binary classification tasks with support for clinical text summarization, dual model pipelines, and rich configuration management.

## Features

- **Multiple Operating Modes**: Standard evaluation, summarization-only, and end-to-end pipelines
- **Clinical Text Summarization**: Specialized engine for processing medical discharge summaries
- **Dual Model Architecture**: Use different models for summarization and evaluation phases
- **High-Performance Inference**: Built on vLLM for efficient GPU utilization and fast inference
- **LoRA Integration**: Utility to merge LoRA adapters with base models for vLLM compatibility
- **Flexible Configuration**: Hydra-powered configuration system with pre-configured workflows
- **Advanced Memory Management**: Automatic GPU memory optimization and model switching
- **Rich UI**: Beautiful console output with progress bars, tables, and formatted results
- **Comprehensive Metrics**: Detailed evaluation metrics including accuracy, precision, recall, F1, ROC-AUC
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

## Operating Modes

LocalLightEval supports three main operating modes:

### 1. Standard Evaluation Mode (Default)
Evaluates text classification performance on pre-processed data:

```bash
# Basic evaluation
python -m locallighteval.main data.input_path=/path/to/your/data.json

# Using pre-configured standard evaluation
python -m locallighteval.main --config-name=standard_eval data.input_path=/path/to/your/data.json
```

### 2. Summarization Mode
Generates summaries from input text (particularly clinical discharge summaries):

```bash
# Clinical summarization
python -m locallighteval.main --config-name=clinical_summarization

# Custom summarization
python -m locallighteval.main mode=summarization data.input_path=/path/to/discharge_summaries.json
```

### 3. End-to-End Mode
Performs summarization followed by evaluation in a single pipeline:

```bash
# End-to-end with single model
python -m locallighteval.main mode=end_to_end data.input_path=/path/to/data.json

# End-to-end with dual models (recommended)
python -m locallighteval.main --config-name=dual_model_pipeline
```

## Clinical Text Summarization

LocalLightEval includes a specialized `ClinicalSummarizationEngine` for processing medical discharge summaries:

### Features
- **Pattern-based extraction** of discharge summary content from complex medical records
- **Template-driven summarization** with clinical-specific prompts
- **Structured output** with summary tags for reliable parsing
- **Error handling** and fallback mechanisms for robust processing

### Usage Examples

```bash
# Clinical summarization with fine-tuned model
python -m locallighteval.main --config-name=clinical_summarization \
    model.name=/path/to/clinical-model \
    data.input_path=/path/to/discharge_summaries.json

# Quick test with small dataset
python -m locallighteval.main --config-name=quick_test \
    mode=summarization \
    data.max_samples=10
```

### Input Data Format
Clinical summarization expects JSON files with discharge summary data:

```json
[
  {
    "text": "Here is the discharge summary:\nName: ___ ... [full discharge summary content] ... Readmission: ...",
    "label": 1
  }
]
```

The system automatically extracts the relevant discharge summary content from the full text.

## Dual Model Pipeline

For optimal performance, use different specialized models for summarization and evaluation:

### Configuration
```yaml
# config/dual_model_pipeline.yaml
dual_model:
  use_different_models: true
  summarization_model:
    name: "meta-llama/Llama-2-7b-chat-hf"
    visible_devices: "0,1"
    gpu_memory_utilization: 0.8
  evaluation_model:
    name: "microsoft/DialoGPT-medium"
    visible_devices: "2"
    gpu_memory_utilization: 0.9
```

### Memory Management
- Automatic model unloading between phases
- GPU memory optimization
- Smart device allocation

## LoRA Model Integration

Use the included merge utility to combine LoRA adapters with base models:

### Merging LoRA Adapters

```bash
# Merge LoRA with base model
python merge_lora.py \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --lora_path /path/to/lora/checkpoint-1600 \
    --output_path /path/to/merged_model

# Use merged model in evaluation
python -m locallighteval.main model.name=/path/to/merged_model
```

### Requirements for LoRA Merging
```bash
pip install peft transformers torch
```

## Pre-configured Workflows

Choose from several pre-configured setups:

### Quick Testing
```bash
python -m locallighteval.main --config-name=quick_test
```
- Small dataset (5 samples)
- Lightweight model
- Debug logging enabled
- Fast inference settings

### Standard Evaluation  
```bash
python -m locallighteval.main --config-name=standard_eval
```
- Full dataset processing
- Evaluation mode only
- Optimized for classification

### Clinical Summarization
```bash
python -m locallighteval.main --config-name=clinical_summarization
```
- Clinical text processing
- Fine-tuned model support
- Medical discharge summary extraction

### Dual Model Pipeline
```bash
python -m locallighteval.main --config-name=dual_model_pipeline
```
- Separate summarization and evaluation models
- Advanced memory management
- Multi-GPU support

## Data Formats

LocalLightEval supports multiple JSON formats:

### JSON Lines Format (.jsonl)
```json
{"text": "This movie is great!", "label": 1}
{"text": "Boring and poorly made.", "label": 0}
```

### JSON Array Format (.json)
```json
[
  {"text": "This movie is great!", "label": 1},
  {"text": "Boring and poorly made.", "label": 0}
]
```

**Required fields:**
- `text`: Input text (string)
- `label`: Binary label (0/1 or boolean)

## Configuration System

### Structure
```
config/
├── config.yaml                    # Main configuration
├── clinical_summarization.yaml    # Clinical workflow
├── dual_model_pipeline.yaml      # Dual model setup
├── quick_test.yaml               # Development testing
├── standard_eval.yaml           # Standard evaluation
├── mode/
│   ├── end_to_end.yaml         # End-to-end processing
│   ├── summarization.yaml      # Summarization only
│   └── evaluation.yaml         # Evaluation only
└── inference/
    └── default.yaml            # Inference parameters
```

### Custom Configurations

```bash
# Override any parameter
python -m locallighteval.main \
    model.name=meta-llama/Llama-2-7b-chat-hf \
    model.visible_devices="0,1" \
    inference.temperature=0.1 \
    data.max_samples=100

# Use custom config file
python -m locallighteval.main --config-name=my_custom_config
```

### Key Configuration Options

```yaml
# Model settings
model:
  name: "microsoft/DialoGPT-medium"
  visible_devices: "0"
  gpu_memory_utilization: 0.9
  max_model_len: 2048

# Operation mode
mode: evaluation  # evaluation, summarization, end_to_end

# Dual model setup (for end_to_end mode)
dual_model:
  use_different_models: true
  summarization_model: {...}
  evaluation_model: {...}

# Summarization settings
summarization:
  output_suffix: "_summaries"
  save_original_text: true
```

## Advanced Features

### GPU Memory Optimization
```bash
# Adjust memory utilization
python -m locallighteval.main model.gpu_memory_utilization=0.7

# Multi-GPU setup
python -m locallighteval.main model.tensor_parallel_size=2 model.visible_devices="0,1"
```

### Logging Control
```bash
# Enable debug logging
python -m locallighteval.main debug=true

# Disable vLLM verbose output
python -m locallighteval.main disable_vllm_logging=true
```

### Batch Processing
```bash
# Optimize batch size
python -m locallighteval.main inference.batch_size=32

# Limit dataset size
python -m locallighteval.main data.max_samples=1000
```

## Output Structure

Each run creates a comprehensive output directory:

```
outputs/eval_20240827_143022/
├── config.yaml                # Complete configuration
├── evaluation.log             # Detailed logs
├── metrics.json               # Evaluation metrics
├── detailed_results.json      # Per-sample results
├── detailed_results.csv       # CSV format
├── predictions.json           # Raw predictions
├── error_analysis.json        # Error patterns
├── run_metadata.json         # System metadata
├── run_summary.txt           # Human-readable summary
└── [dataset]_summaries.json  # Generated summaries (if applicable)
```

## Performance Optimization

### GPU Configuration
- **Memory**: Adjust `gpu_memory_utilization` (0.7-0.95)
- **Multi-GPU**: Use `tensor_parallel_size` for model parallelism
- **Device Selection**: Set `visible_devices` for specific GPUs

### Processing Efficiency
- **Batch Size**: Increase for better throughput (monitor GPU memory)
- **Model Size**: Use appropriate model size for your hardware
- **Data Limiting**: Use `max_samples` for testing and development

### Memory Management
- **Automatic Cleanup**: Models are automatically unloaded in dual-model mode
- **Context Length**: Set `max_model_len` to limit memory usage
- **Precision**: Use appropriate `dtype` (auto, float16, bfloat16)

## Troubleshooting

### Common Issues

**GPU Memory Errors**:
```bash
# Reduce memory usage
python -m locallighteval.main model.gpu_memory_utilization=0.7 inference.batch_size=8
```

**Model Loading Issues**:
```bash
# For models requiring trust_remote_code
python -m locallighteval.main model.trust_remote_code=true

# Check model path
python -m locallighteval.main model.name=/absolute/path/to/model
```

**LoRA Merging Errors**:
```bash
# Install required packages
pip install peft transformers torch

# Check adapter path
python merge_lora.py --lora_path /path/to/adapter/checkpoint-1600
```

**Configuration Issues**:
```bash
# Validate config only
python -m locallighteval.main dry_run=true

# Debug mode for detailed logs
python -m locallighteval.main debug=true
```

### Mode-Specific Issues

**Summarization Mode**:
- Ensure input data contains discharge summary text
- Check pattern matching in extraction logs
- Verify model can handle medical terminology

**Dual Model Mode**:
- Ensure sufficient GPU memory for both models
- Check device allocation doesn't conflict
- Monitor GPU memory during model switching

**End-to-End Mode**:
- Verify both summarization and evaluation work independently
- Check intermediate summary quality
- Monitor total processing time

## Development and Testing

### Quick Development Workflow
```bash
# Test with minimal dataset
python -m locallighteval.main --config-name=quick_test dry_run=true

# Run actual quick test
python -m locallighteval.main --config-name=quick_test data.input_path=/path/to/small_data.json
```

### Environment Validation
```bash
# Check dependencies and GPU
python -m locallighteval.main dry_run=true
```

### Example Generation
```bash
# Generate sample data
python example_run.py
```

## Available Models

### Recommended Models by Use Case

**Clinical Summarization**:
- Fine-tuned medical models (Qwen3-4B-clinical, Bio-LLaMA)
- General instruction-tuned models (Llama-2-7b-chat, Mistral-7B-Instruct)

**Text Classification**:
- Conversation models (DialoGPT-medium, DialoGPT-large)
- General LLMs (Llama-2, Mistral, CodeLlama)

**Usage Examples**:
```bash
# Medical models
python -m locallighteval.main model.name=/ssd-shared/qwen/qwen3-4b-clinical-RL-tuned

# HuggingFace models
python -m locallighteval.main model.name=meta-llama/Llama-2-7b-chat-hf
python -m locallighteval.main model.name=mistralai/Mistral-7B-Instruct-v0.1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

For clinical summarization features, please include medical domain expertise validation.

## Acknowledgments

- Built on [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- Uses [Hydra](https://github.com/facebookresearch/hydra) for configuration management
- Powered by [Rich](https://github.com/Textualize/rich) for console output
- Leverages [HuggingFace Transformers](https://github.com/huggingface/transformers) ecosystem
- [PEFT](https://github.com/huggingface/peft) for LoRA integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.