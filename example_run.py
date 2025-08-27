#!/usr/bin/env python3
"""Example script to run LocalLightEval."""

import json
import tempfile
from pathlib import Path

def create_sample_data():
    """Create sample data for testing."""
    sample_data = [
        {"text": "This movie is absolutely fantastic! I loved every minute of it.", "label": 1},
        {"text": "Terrible film, completely boring and poorly acted.", "label": 0},
        {"text": "Great acting, wonderful story, highly recommend!", "label": 1},
        {"text": "Waste of time, very disappointing experience.", "label": 0},
        {"text": "Amazing cinematography and compelling characters.", "label": 1},
        {"text": "Poorly written script and bad direction.", "label": 0},
        {"text": "One of the best movies I've seen this year!", "label": 1},
        {"text": "Boring plot with no character development.", "label": 0},
        {"text": "Excellent performances by all the actors.", "label": 1},
        {"text": "Confusing storyline and weak ending.", "label": 0},
    ]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    for item in sample_data:
        json.dump(item, temp_file)
        temp_file.write('\n')
    
    temp_file.close()
    return temp_file.name


if __name__ == "__main__":
    print("Creating sample data...")
    sample_file = create_sample_data()
    print(f"Sample data created at: {sample_file}")
    
    print("\nTo run the evaluation with this sample data:")
    print(f"python -m locallighteval.main data.input_path={sample_file}")
    
    print("\nOr with different models (any HuggingFace model or local path):")
    print(f"python -m locallighteval.main data.input_path={sample_file} model.name=meta-llama/Llama-2-7b-chat-hf")
    print(f"python -m locallighteval.main data.input_path={sample_file} model.name=mistralai/Mistral-7B-Instruct-v0.1")
    print(f"python -m locallighteval.main data.input_path={sample_file} model.name=/path/to/local/model")
    
    print("\nOr with few-shot prompting:")
    print(f"python -m locallighteval.main data.input_path={sample_file} inference=few_shot")
    
    print("\nCombining model and inference settings:")
    print(f"python -m locallighteval.main data.input_path={sample_file} model.name=meta-llama/Llama-2-7b-chat-hf inference=few_shot")
    
    print("\nWith custom model parameters:")
    print(f"python -m locallighteval.main data.input_path={sample_file} \\")
    print(f"    model.name=mistralai/Mistral-7B-Instruct-v0.1 \\")
    print(f"    model.tensor_parallel_size=2 \\")
    print(f"    model.gpu_memory_utilization=0.8 \\")
    print(f"    inference.batch_size=64")
    
    # Clean up
    import atexit
    import os
    atexit.register(lambda: os.unlink(sample_file))