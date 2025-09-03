"""Utility functions for LocalLightEval."""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.text import Text
from rich.layout import Layout
from rich.live import Live


console = Console()


def setup_rich_logging(output_dir: Path, log_level: str = "INFO") -> None:
    """Setup Rich-enhanced logging configuration."""
    logger.remove()
    
    # Rich console logging
    logger.add(
        RichHandler(console=console, rich_tracebacks=True, show_path=False),
        level=log_level,
        format="{message}",
        backtrace=True,
        diagnose=True
    )
    
    # File logging
    log_file = output_dir / "evaluation.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="7 days"
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def print_banner() -> None:
    """Print a welcome banner using Rich."""
    banner_text = """
    ██╗     ██████╗  ██████╗ █████╗ ██╗         ██╗     ██╗ ██████╗ ██╗  ██╗████████╗    ███████╗██╗   ██╗ █████╗ ██╗     
    ██║    ██╔═══██╗██╔════╝██╔══██╗██║         ██║     ██║██╔════╝ ██║  ██║╚══██╔══╝    ██╔════╝██║   ██║██╔══██╗██║     
    ██║    ██║   ██║██║     ███████║██║         ██║     ██║██║  ███╗███████║   ██║       █████╗  ██║   ██║███████║██║     
    ██║    ██║   ██║██║     ██╔══██║██║         ██║     ██║██║   ██║██╔══██║   ██║       ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     
    ███████╗██████╔╝╚██████╗██║  ██║███████╗    ███████╗██║╚██████╔╝██║  ██║   ██║       ███████╗ ╚████╔╝ ██║  ██║███████╗
    ╚══════╝╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
    """
    
    panel = Panel(
        Text(banner_text, style="bold blue"),
        title="[bold white]LLM Evaluation Benchmark",
        subtitle="[italic]Powered by vLLM and HuggingFace Transformers",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)


def print_config_summary(cfg) -> None:
    """Print a formatted summary of the configuration."""
    table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="bold cyan", width=15)
    table.add_column("Setting", style="yellow", width=20)
    table.add_column("Value", style="green")
    
    # Model configuration
    table.add_row("Model", "Name", str(cfg.model.name))
    table.add_row("", "Tensor Parallel", str(cfg.model.tensor_parallel_size))
    table.add_row("", "GPU Memory Util", f"{cfg.model.gpu_memory_utilization:.1%}")
    table.add_row("", "Max Model Length", str(cfg.model.max_model_len or "Auto"))
    
    # Inference configuration
    table.add_row("Inference", "Max Tokens", str(cfg.inference.max_tokens))
    table.add_row("", "Temperature", str(cfg.inference.temperature))
    table.add_row("", "Batch Size", str(cfg.inference.batch_size))
    
    # Data configuration
    table.add_row("Data", "Input Path", str(cfg.data.input_path))
    table.add_row("", "Text Key", cfg.data.text_key)
    table.add_row("", "Label Key", cfg.data.label_key)
    table.add_row("", "Max Samples", str(cfg.data.max_samples or "All"))
    
    # Output configuration
    table.add_row("Output", "Output Directory", str(Path.cwd()))
    table.add_row("", "Save Predictions", "✓" if cfg.output.save_predictions else "✗")
    table.add_row("", "Save Metrics", "✓" if cfg.output.save_metrics else "✗")
    
    console.print(table)


def create_progress_bar() -> Progress:
    """Create a Rich progress bar for tracking evaluation."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )


def print_gpu_info() -> None:
    """Print GPU information using Rich."""
    gpu_info = check_gpu_availability()
    
    if gpu_info["cuda_available"]:
        table = Table(title="GPU Information", show_header=True, header_style="bold green")
        table.add_column("Device", style="cyan")
        table.add_column("Name", style="yellow") 
        table.add_column("Total Memory", style="magenta")
        table.add_column("Allocated", style="red")
        table.add_column("Cached", style="blue")
        
        for gpu in gpu_info["gpu_memory"]:
            table.add_row(
                f"cuda:{gpu['device_id']}",
                gpu["name"],
                f"{gpu['total_memory_gb']:.1f} GB",
                f"{gpu['allocated_memory_gb']:.1f} GB", 
                f"{gpu['cached_memory_gb']:.1f} GB"
            )
        
        console.print(table)
    else:
        console.print(Panel(
            "[red]No CUDA GPUs available[/red]",
            title="GPU Status",
            border_style="red"
        ))


def save_run_metadata(
    output_dir: Path, 
    config: Dict[str, Any], 
    start_time: datetime,
    end_time: datetime = None,
    additional_info: Dict[str, Any] = None
) -> None:
    """Save metadata about the evaluation run."""
    end_time = end_time or datetime.now()
    
    metadata = {
        "run_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "output_directory": str(output_dir),
        },
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd(),
        },
        "config": config
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    metadata_file = output_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Run metadata saved to {metadata_file}")


def create_run_summary(
    output_dir: Path,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    sample_count: int,
    duration: float
) -> None:
    """Create a human-readable run summary."""
    summary_file = output_dir / "run_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LOCAL LIGHT EVAL - RUN SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Run Directory: {output_dir}\n")
        f.write(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration:.2f} seconds\n")
        f.write(f"Total Samples: {sample_count}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model: {config.get('model', {}).get('name', 'Unknown')}\n")
        f.write(f"Max Tokens: {config.get('inference', {}).get('max_tokens', 'Unknown')}\n")
        f.write(f"Temperature: {config.get('inference', {}).get('temperature', 'Unknown')}\n")
        f.write(f"Batch Size: {config.get('inference', {}).get('batch_size', 'Unknown')}\n\n")
        
        f.write("DATA CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Path: {config.get('data', {}).get('input_path', 'Unknown')}\n")
        f.write(f"Text Key: {config.get('data', {}).get('text_key', 'Unknown')}\n")
        f.write(f"Label Key: {config.get('data', {}).get('label_key', 'Unknown')}\n\n")
        
        if metrics:
            f.write("EVALUATION RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy:     {metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Precision:    {metrics.get('precision', 'N/A'):.4f}\n")
            f.write(f"Recall:       {metrics.get('recall', 'N/A'):.4f}\n")
            f.write(f"F1-Score:     {metrics.get('f1_score', 'N/A'):.4f}\n")
            
            if metrics.get('roc_auc') is not None:
                f.write(f"ROC-AUC:      {metrics['roc_auc']:.4f}\n")
            
            f.write(f"\nValid Predictions: {metrics.get('valid_predictions', 'N/A')}\n")
            f.write(f"Invalid Predictions: {metrics.get('invalid_predictions', 'N/A')}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  TN: {metrics.get('true_negatives', 'N/A'):>6}  FP: {metrics.get('false_positives', 'N/A'):>6}\n")
            f.write(f"  FN: {metrics.get('false_negatives', 'N/A'):>6}  TP: {metrics.get('true_positives', 'N/A'):>6}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"Run summary saved to {summary_file}")


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and memory."""
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_memory": []
    }
    
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        
        if gpu_info["cuda_available"]:
            gpu_info["gpu_count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["gpu_count"]):
                memory_info = {
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                    "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1e9,
                    "cached_memory_gb": torch.cuda.memory_reserved(i) / 1e9
                }
                gpu_info["gpu_memory"].append(memory_info)
    
    except ImportError:
        logger.warning("PyTorch not available, cannot check GPU status")
    except Exception as e:
        logger.warning(f"Error checking GPU status: {e}")
    
    return gpu_info


def validate_environment() -> Dict[str, bool]:
    """Validate that the environment has required dependencies."""
    validation_results = {}
    
    required_packages = [
        "vllm",
        "transformers", 
        "torch",
        "sklearn",
        "pandas",
        "numpy",
        "pydantic",
        "yaml",
        "tqdm",
        "loguru"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            validation_results[package] = True
        except ImportError:
            validation_results[package] = False
            logger.error(f"Required package '{package}' is not installed")
    
    return validation_results


def estimate_memory_requirements(
    model_name: str, 
    batch_size: int,
    sequence_length: int = 2048
) -> Dict[str, float]:
    """Rough estimation of memory requirements."""
    
    # Very rough estimates based on common models
    model_size_estimates = {
        "gpt2": 0.5,
        "gpt2-medium": 1.4,
        "gpt2-large": 3.0,
        "gpt2-xl": 6.0,
        "llama-7b": 14.0,
        "llama-13b": 26.0,
        "llama-30b": 60.0,
        "llama-65b": 130.0,
    }
    
    # Try to estimate based on model name
    estimated_model_gb = 7.0  # Default assumption
    for name, size in model_size_estimates.items():
        if name.lower() in model_name.lower():
            estimated_model_gb = size
            break
    
    # Rough estimates for inference memory
    sequence_memory_gb = (batch_size * sequence_length * 4) / 1e9  # 4 bytes per float32
    activation_memory_gb = estimated_model_gb * 0.2  # Rough estimate
    total_estimated_gb = estimated_model_gb + sequence_memory_gb + activation_memory_gb
    
    return {
        "estimated_model_size_gb": estimated_model_gb,
        "estimated_sequence_memory_gb": sequence_memory_gb,
        "estimated_activation_memory_gb": activation_memory_gb,
        "estimated_total_gb": total_estimated_gb
    }