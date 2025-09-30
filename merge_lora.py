#!/usr/bin/env python3
"""
Script to merge LoRA adapter with base model for vLLM compatibility.

Usage:
    python merge_lora.py --base_model /path/to/local/base/model \
                        --lora_path /ssd-shared/ryans_output/random_text_no_lower_limit/checkpoint-1600 \
                        --output_path /ssd-shared/merged_models/qwen3-4b-clinical-merged
"""

import argparse
import os
from pathlib import Path

def merge_lora_adapter(base_model: str, lora_path: str, output_path: str):
    """Merge LoRA adapter with base model."""
    try:
        from peft import AutoPeftModelForCausalLM, PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"Loading base model from local path: {base_model}")
        print(f"Loading LoRA adapter from: {lora_path}")

        # First load the base model from local path
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True  # Force local loading
        )

        # Load the PEFT model with the explicit base model
        model = PeftModel.from_pretrained(
            base_model_obj,
            lora_path,
            torch_dtype=torch.float16
        )
        
        # Try to load tokenizer from LoRA path first, fallback to base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(lora_path)
        except:
            print("Tokenizer not found in LoRA path, loading from base model...")
            tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        
        print("Merging LoRA adapter with base model...")
        # Merge the adapter weights into the base model
        merged_model = model.merge_and_unload()
        
        print(f"Saving merged model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        
        # Save the merged model
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("✅ LoRA merge completed successfully!")
        print(f"You can now use: {output_path}")
        
        return output_path
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install: pip install peft transformers torch")
        return None
    except Exception as e:
        print(f"❌ Error merging LoRA: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507", help="Local path to base model directory")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_path", required=True, help="Output path for merged model")
    
    args = parser.parse_args()
    
    merge_lora_adapter(args.base_model, args.lora_path, args.output_path)

if __name__ == "__main__":
    main()