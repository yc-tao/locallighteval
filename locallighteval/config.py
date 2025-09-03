"""Configuration management for LocalLightEval using Hydra."""

from datetime import datetime
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    name: str
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    visible_devices: Optional[str] = None  # e.g., "0,1" or "2"


@dataclass  
class InferenceConfig:
    """Configuration for inference parameters."""
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    batch_size: int = 32
    instruction: str = "Classify the following text as positive (1) or negative (0). Respond with only the number."


@dataclass
class DataConfig:
    """Configuration for data loading."""
    input_path: str
    text_key: str = "text"
    label_key: str = "label"
    max_samples: Optional[int] = None


@dataclass
class OutputConfig:
    """Configuration for output management."""
    save_predictions: bool = True
    save_metrics: bool = True
    save_detailed_results: bool = True


@dataclass
class SummarizationConfig:
    """Configuration for summarization settings."""
    output_suffix: str = "_summaries"
    extract_patterns: bool = True
    save_original_text: bool = True


@dataclass 
class DualModelConfig:
    """Configuration for dual model setup in end-to-end mode."""
    use_different_models: bool = False
    summarization_model: ModelConfig = None
    evaluation_model: ModelConfig = None


def validate_config(cfg: DictConfig) -> None:
    """Validate the configuration."""
    required_fields = {
        "model.name": "Model name is required",
        "data.input_path": "Input data path is required",
    }
    
    for field_path, error_msg in required_fields.items():
        if not OmegaConf.select(cfg, field_path):
            raise ValueError(f"Configuration error: {error_msg}")
    
    # Validate file exists
    if not Path(cfg.data.input_path).exists():
        raise FileNotFoundError(f"Input file not found: {cfg.data.input_path}")
    
    # Validate numeric ranges
    if cfg.inference.temperature < 0:
        raise ValueError("Temperature must be >= 0")
    
    if cfg.inference.top_p <= 0 or cfg.inference.top_p > 1:
        raise ValueError("Top-p must be in (0, 1]")
    
    if cfg.inference.batch_size <= 0:
        raise ValueError("Batch size must be > 0")
    
    if cfg.model.gpu_memory_utilization <= 0 or cfg.model.gpu_memory_utilization > 1:
        raise ValueError("GPU memory utilization must be in (0, 1]")