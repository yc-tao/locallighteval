"""Inference engine using vLLM backend."""

import os
import torch
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from vllm import LLM, SamplingParams
from loguru import logger
from tqdm import tqdm

from .config import ModelConfig, InferenceConfig


class VLLMInferenceEngine:
    """Inference engine using vLLM for efficient LLM inference."""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        self.llm = None
        self.sampling_params = None
        
        self._initialize_model()
        self._setup_sampling_params()
    
    def _validate_model_path(self) -> None:
        """Validate model path before loading."""
        model_path = self.model_config.name
        
        # Check if it's a local path
        if os.path.exists(model_path):
            path_obj = Path(model_path)
            if path_obj.is_dir():
                # Check for required model files in local directory
                required_files = ['config.json']
                missing_files = [f for f in required_files if not (path_obj / f).exists()]
                if missing_files:
                    logger.warning(f"Local model directory missing files: {missing_files}")
                    logger.info("Proceeding with load attempt - vLLM may handle missing files")
                logger.info(f"Using local model directory: {model_path}")
            else:
                raise ValueError(f"Model path is not a directory: {model_path}")
        else:
            # Assume it's a HuggingFace repo name
            logger.info(f"Using HuggingFace model repository: {model_path}")
            
            # Basic validation of HF repo name format
            if not model_path or '/' not in model_path:
                logger.warning(f"Model name '{model_path}' doesn't follow typical HF format (org/model)")
                logger.info("Proceeding with load attempt - may be a valid model name")

    def _initialize_model(self) -> None:
        """Initialize the vLLM model."""
        logger.info(f"Initializing vLLM model: {self.model_config.name}")
        
        # Validate model path/name
        self._validate_model_path()
        
        # Set GPU visibility if specified
        if self.model_config.visible_devices is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.model_config.visible_devices)
            logger.info(f"Set CUDA_VISIBLE_DEVICES to: {self.model_config.visible_devices}")
        
        
        try:
            # Build vLLM arguments
            vllm_args = {
                "model": self.model_config.name,
                "trust_remote_code": self.model_config.trust_remote_code,
                "tensor_parallel_size": self.model_config.tensor_parallel_size,
                "dtype": self.model_config.dtype,
                "gpu_memory_utilization": self.model_config.gpu_memory_utilization,
                "disable_log_stats": True,  # Reduce vLLM's verbose statistics logging
            }
            
            # Only add max_model_len if specified
            if self.model_config.max_model_len is not None:
                vllm_args["max_model_len"] = self.model_config.max_model_len
            
            logger.debug(f"vLLM initialization arguments: {vllm_args}")
            
            self.llm = LLM(**vllm_args)
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model '{self.model_config.name}': {e}")
            
            # Provide helpful error messages
            if "not found" in str(e).lower():
                logger.error("Model not found. Please check:")
                logger.error("1. For HuggingFace models: Verify the model name exists on https://huggingface.co")
                logger.error("2. For local models: Ensure the path exists and contains model files")
                logger.error("3. Check your internet connection for downloading HF models")
            elif "memory" in str(e).lower():
                logger.error("Out of memory. Try:")
                logger.error("1. Reducing gpu_memory_utilization (current: {:.1f})".format(self.model_config.gpu_memory_utilization))
                logger.error("2. Using a smaller model")
                logger.error("3. Reducing max_model_len if set")
            elif "trust_remote_code" in str(e).lower():
                logger.error("Model requires trust_remote_code=true. Set this in your config if you trust the model code.")
            
            raise
    
    def _setup_sampling_params(self) -> None:
        """Setup sampling parameters for inference."""
        # Calculate max prompt tokens if truncation is enabled
        truncate_prompt_tokens = None
        if self.inference_config.truncate_input:
            model_max = self.model_config.max_model_len or 2048  # Default fallback
            output_tokens = self.inference_config.max_tokens
            buffer = 50  # Safety buffer for special tokens
            truncate_prompt_tokens = model_max - output_tokens - buffer
        
        self.sampling_params = SamplingParams(
            max_tokens=self.inference_config.max_tokens,
            temperature=self.inference_config.temperature,
            top_p=self.inference_config.top_p,
            top_k=self.inference_config.top_k,
            truncate_prompt_tokens=truncate_prompt_tokens,
        )
        
        logger.info(f"Sampling parameters: {self.sampling_params}")
    

    def cleanup(self) -> None:
        """Clean up model resources and free GPU memory."""
        if hasattr(self, 'llm') and self.llm is not None:
            logger.info("Cleaning up vLLM model...")
            del self.llm
            self.llm = None
            
            # Force GPU memory cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        self.cleanup()
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a batch of prompts."""
        if not prompts:
            return []
        
        logger.debug(f"Generating responses for {len(prompts)} prompts")
        
        try:
            # Override sampling params if provided
            sampling_params = self.sampling_params
            if kwargs:
                # Calculate truncate_prompt_tokens for kwargs
                truncate_prompt_tokens = None
                if self.inference_config.truncate_input:
                    model_max = self.model_config.max_model_len or 2048
                    output_tokens = kwargs.get('max_tokens', self.inference_config.max_tokens)
                    buffer = 50
                    truncate_prompt_tokens = model_max - output_tokens - buffer
                
                sampling_params = SamplingParams(
                    max_tokens=kwargs.get('max_tokens', self.inference_config.max_tokens),
                    temperature=kwargs.get('temperature', self.inference_config.temperature),
                    top_p=kwargs.get('top_p', self.inference_config.top_p),
                    top_k=kwargs.get('top_k', self.inference_config.top_k),
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            responses = []
            for output in outputs:
                generated_text = output.outputs[0].text
                responses.append(generated_text)
            
            return responses
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def generate_with_metadata(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses with additional metadata."""
        if not prompts:
            return []
        
        logger.debug(f"Generating responses with metadata for {len(prompts)} prompts")
        
        try:
            sampling_params = self.sampling_params
            if kwargs:
                # Calculate truncate_prompt_tokens for kwargs
                truncate_prompt_tokens = None
                if self.inference_config.truncate_input:
                    model_max = self.model_config.max_model_len or 2048
                    output_tokens = kwargs.get('max_tokens', self.inference_config.max_tokens)
                    buffer = 50
                    truncate_prompt_tokens = model_max - output_tokens - buffer
                
                sampling_params = SamplingParams(
                    max_tokens=kwargs.get('max_tokens', self.inference_config.max_tokens),
                    temperature=kwargs.get('temperature', self.inference_config.temperature),
                    top_p=kwargs.get('top_p', self.inference_config.top_p),
                    top_k=kwargs.get('top_k', self.inference_config.top_k),
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            results = []
            for i, output in enumerate(outputs):
                # Check if truncation occurred by comparing token counts
                expected_tokens = len(output.prompt_token_ids)
                max_prompt_tokens = sampling_params.truncate_prompt_tokens
                prompt_truncated = max_prompt_tokens is not None and expected_tokens >= max_prompt_tokens
                
                result = {
                    "prompt": prompts[i],  # Original prompt
                    "generated_text": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason,
                    "prompt_truncated": prompt_truncated,  # Flag if truncation likely occurred
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during generation with metadata: {e}")
            raise


class BinaryClassificationInference:
    """Specialized inference for binary classification tasks."""
    
    def __init__(self, inference_engine: VLLMInferenceEngine):
        self.inference_engine = inference_engine
    
    def create_classification_prompt(
        self, 
        text: str, 
        instruction: str = "Classify the following text as positive (1) or negative (0). Respond with only the number.",
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Create a prompt for binary classification."""
        prompt = instruction + "\n\n"
        
        if few_shot_examples:
            for example in few_shot_examples:
                prompt += f"Text: {example['text']}\nLabel: {example['label']}\n\n"
        
        prompt += f"Text: {text}\nLabel:"
        
        return prompt
    
    def classify_batch(
        self, 
        texts: List[str],
        instruction: str = "Classify the following text as positive (1) or negative (0). Respond with only the number.",
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Classify a batch of texts."""
        logger.info(f"Classifying {len(texts)} texts")
        
        # Create prompts
        prompts = [
            self.create_classification_prompt(text, instruction, few_shot_examples) 
            for text in texts
        ]
        
        # Generate responses with metadata
        results = self.inference_engine.generate_with_metadata(prompts)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            processed_result = {
                "input_text": texts[i],
                "prompt": result["prompt"],
                "raw_response": result["generated_text"],
                "predicted_label": self._parse_classification_response(result["generated_text"]),
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "finish_reason": result["finish_reason"]
            }
            processed_results.append(processed_result)
        
        return processed_results
    
    def _parse_classification_response(self, response: str) -> Optional[int]:
        """Parse the model's response to extract classification label."""
        response = response.strip().lower()
        
        # Look for explicit 0 or 1
        if response == "0":
            return 0
        elif response == "1":
            return 1
        
        # Look for 0 or 1 at the beginning of response
        if response.startswith("0"):
            return 0
        elif response.startswith("1"):
            return 1
        
        # Look for words
        if "positive" in response or "yes" in response or "true" in response:
            return 1
        elif "negative" in response or "no" in response or "false" in response:
            return 0
        
        # If we can't parse, return None
        logger.warning(f"Could not parse classification response: '{response}'")
        return None