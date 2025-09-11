"""Text summarization module for generating clinical summaries."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from rich.progress import Progress, TaskID

from .inference import VLLMInferenceEngine


class ClinicalSummarizationEngine:
    """Engine for generating clinical summaries from discharge summaries."""
    
    def __init__(self, inference_engine: VLLMInferenceEngine, debug: bool = False):
        """Initialize the summarization engine.
        
        Args:
            inference_engine: The vLLM inference engine to use for generation
            debug: If True, print raw LLM responses for debugging
        """
        self.inference_engine = inference_engine
        self.debug = debug
        self.prompt_template = (
            "{full_note}\n\n"
            "You are a clinical summarization assistant. Extract a short summary of this discharge summary between <summary> and </summary>. Be concise and to the point:"
        )
    
    def extract_discharge_summary(self, full_text: str) -> str:
        """Extract the discharge summary from the full text.
        
        Args:
            full_text: The complete text containing the discharge summary
            
        Returns:
            The extracted discharge summary text
        """
        # Look for patterns that indicate the start and end of discharge summary
        # The current data seems to contain a lot of prompt text before the actual discharge summary
        
        # Try to find the discharge summary section
        patterns = [
            r"Here is the discharge summary:\s*(.*?)(?:\n\s*Readmission:|$)",
            r"discharge summary:\s*(.*?)(?:\n\s*Readmission:|$)",
            r"Name:\s+___.*?(?:\n\s*Readmission:|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                if len(summary) > 100:  # Ensure we got a substantial summary
                    return summary
        
        # If no pattern matches, return the full text (fallback)
        logger.warning("Could not extract discharge summary using patterns, using full text")
        return full_text
    
    def generate_summary(self, full_note: str) -> str:
        """Generate a clinical summary for a full note.
        
        Args:
            full_note: The full note text from the JSON data
            
        Returns:
            Generated clinical summary
        """
        # Format the prompt
        prompt = self.prompt_template.format(full_note=full_note)
        
        # Generate response using the inference engine
        responses = self.inference_engine.generate([prompt])
        
        if not responses or not responses[0]:
            logger.error("No response generated from inference engine")
            return ""
        
        generated_text = responses[0]
        
        # Print raw LLM response (generated tokens only) if debug mode is enabled
        if self.debug:
            print(f"\n{'='*80}")
            print("LLM GENERATED TOKENS:")
            print(f"{'='*80}")
            print(generated_text)
            print(f"{'='*80}\n")
        
        # Extract text between <summary> and </summary> tags
        summary_match = re.search(r'<summary>(.*?)</summary>', generated_text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
            logger.debug(f"Extracted summary: {summary[:100]}...")
            return summary
        else:
            # If no tags found, return the generated text as is (but log warning)
            logger.warning("Generated text does not contain <summary> tags, using raw output")
            return generated_text.strip()
    
    def process_dataset(self, input_data: List[Dict[str, Any]], 
                       progress: Optional[Progress] = None,
                       task_id: Optional[TaskID] = None) -> List[Dict[str, Any]]:
        """Process a dataset to generate summaries for all entries.
        
        Args:
            input_data: List of input data entries
            progress: Optional Rich progress bar
            task_id: Optional task ID for progress tracking
            
        Returns:
            List of processed entries with generated summaries
        """
        processed_data = []
        
        for i, entry in enumerate(input_data):
            # Generate summary using the full note directly
            try:
                summary = self.generate_summary(entry['text'])
                
                # Create new entry with the summary as 'text' and preserve other fields
                processed_entry = {
                    'text': summary,
                    'label': entry['label'],
                    'original_text': entry['text'],  # Keep original for reference
                }
                
                # Preserve other fields if they exist
                for key, value in entry.items():
                    if key not in ['text', 'label']:
                        processed_entry[key] = value
                
                processed_data.append(processed_entry)
                
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
                
                logger.debug(f"Processed entry {i+1}/{len(input_data)}")
                
            except Exception as e:
                logger.error(f"Failed to process entry {i+1}: {e}")
                # Create entry with empty summary to maintain data integrity
                processed_entry = {
                    'text': "",
                    'label': entry['label'],
                    'original_text': entry['text'],
                    'error': str(e)
                }
                processed_data.append(processed_entry)
                
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)
        
        return processed_data
    
    def save_summaries(self, processed_data: List[Dict[str, Any]], 
                      output_path: Path) -> None:
        """Save processed summaries to a JSON file.
        
        Args:
            processed_data: List of processed entries
            output_path: Path to save the output file
        """
        try:
            with open(output_path, 'w') as f:
                for entry in processed_data:
                    json.dump(entry, f)
                    f.write('\n')
            
            logger.info(f"Saved {len(processed_data)} summaries to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save summaries to {output_path}: {e}")
            raise