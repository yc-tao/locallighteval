"""Inference engine using vLLM backend."""

import torch
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
    
    def _initialize_model(self) -> None:
        """Initialize the vLLM model."""
        logger.info(f"Initializing vLLM model: {self.model_config.name}")
        
        try:
            self.llm = LLM(
                model=self.model_config.name,
                trust_remote_code=self.model_config.trust_remote_code,
                tensor_parallel_size=self.model_config.tensor_parallel_size,
                dtype=self.model_config.dtype,
                max_model_len=self.model_config.max_model_len,
                gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _setup_sampling_params(self) -> None:
        """Setup sampling parameters for inference."""
        self.sampling_params = SamplingParams(
            max_tokens=self.inference_config.max_tokens,
            temperature=self.inference_config.temperature,
            top_p=self.inference_config.top_p,
            top_k=self.inference_config.top_k,
        )
        
        logger.info(f"Sampling parameters: {self.sampling_params}")
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a batch of prompts."""
        if not prompts:
            return []
        
        logger.debug(f"Generating responses for {len(prompts)} prompts")
        
        try:
            # Override sampling params if provided
            sampling_params = self.sampling_params
            if kwargs:
                sampling_params = SamplingParams(
                    max_tokens=kwargs.get('max_tokens', self.inference_config.max_tokens),
                    temperature=kwargs.get('temperature', self.inference_config.temperature),
                    top_p=kwargs.get('top_p', self.inference_config.top_p),
                    top_k=kwargs.get('top_k', self.inference_config.top_k),
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
                sampling_params = SamplingParams(
                    max_tokens=kwargs.get('max_tokens', self.inference_config.max_tokens),
                    temperature=kwargs.get('temperature', self.inference_config.temperature),
                    top_p=kwargs.get('top_p', self.inference_config.top_p),
                    top_k=kwargs.get('top_k', self.inference_config.top_k),
                )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            results = []
            for i, output in enumerate(outputs):
                result = {
                    "prompt": prompts[i],
                    "generated_text": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason,
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