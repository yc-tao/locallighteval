"""Text summarization module for generating clinical summaries."""

import json
import re
import time
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
        """Generate a clinical summary for a single full note.

        Args:
            full_note: The full note text from the JSON data

        Returns:
            Generated clinical summary
        """
        summaries = self.generate_summary_batch([full_note])
        return summaries[0] if summaries else ""

    def generate_summary_batch(self, full_notes: List[str]) -> List[str]:
        """Generate clinical summaries for a batch of full notes.

        Args:
            full_notes: List of full note texts from the JSON data

        Returns:
            List of generated clinical summaries
        """
        if not full_notes:
            return []

        # Format the prompts
        prompts = [self.prompt_template.format(full_note=note) for note in full_notes]

        # Generate responses using the inference engine batch processing
        try:
            responses = self.inference_engine.generate(prompts)
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [""] * len(full_notes)

        if not responses or len(responses) != len(full_notes):
            logger.error(f"Expected {len(full_notes)} responses, got {len(responses) if responses else 0}")
            return [""] * len(full_notes)

        summaries = []
        for i, generated_text in enumerate(responses):
            if not generated_text:
                logger.warning(f"No response generated for batch item {i+1}")
                summaries.append("")
                continue

            # Print raw LLM response (generated tokens only) if debug mode is enabled
            if self.debug:
                print(f"\n{'='*80}")
                print(f"LLM GENERATED TOKENS (batch item {i+1}):")
                print(f"{'='*80}")
                print(generated_text)
                print(f"{'='*80}\n")

            # Extract text between <summary> and </summary> tags
            summary_match = re.search(r'<summary>(.*?)</summary>', generated_text, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1).strip()
                logger.debug(f"Extracted summary for item {i+1}: {summary[:100]}...")
                summaries.append(summary)
            else:
                # If no tags found, return the generated text as is (but log warning)
                logger.warning(f"Generated text for item {i+1} does not contain <summary> tags, using raw output")
                summaries.append(generated_text.strip())

        return summaries
    
    def process_dataset(self, input_data: List[Dict[str, Any]],
                       batch_size: int = 1,
                       output_path: Optional[Path] = None,
                       progress: Optional[Progress] = None,
                       task_id: Optional[TaskID] = None) -> List[Dict[str, Any]]:
        """Process a dataset to generate summaries for all entries using batch processing.

        Args:
            input_data: List of input data entries
            batch_size: Number of entries to process in each batch
            output_path: Optional path to save results incrementally
            progress: Optional Rich progress bar
            task_id: Optional task ID for progress tracking

        Returns:
            List of processed entries with generated summaries
        """
        processed_data = []
        total_entries = len(input_data)
        total_start_time = time.time()

        # Process data in batches
        for batch_start in range(0, total_entries, batch_size):
            batch_end = min(batch_start + batch_size, total_entries)
            batch_entries = input_data[batch_start:batch_end]
            batch_texts = [entry['text'] for entry in batch_entries]

            batch_num = batch_start//batch_size + 1
            total_batches = (total_entries + batch_size - 1)//batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}, entries {batch_start+1}-{batch_end}")

            try:
                # Time the batch processing
                batch_start_time = time.time()

                # Generate summaries for the batch
                batch_summaries = self.generate_summary_batch(batch_texts)

                # Calculate and log timing
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                entries_per_second = len(batch_texts) / batch_duration if batch_duration > 0 else 0

                logger.info(f"Batch {batch_num} completed in {batch_duration:.2f}s ({entries_per_second:.1f} entries/sec)")

                # Process each entry in the batch
                for i, (entry, summary) in enumerate(zip(batch_entries, batch_summaries)):
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

                # Save incremental results if output path is provided
                if output_path:
                    self._save_batch_results(processed_data[batch_start:], output_path, append=(batch_start > 0))

                logger.debug(f"Successfully processed batch {batch_num} ({batch_start+1}-{batch_end})")

            except Exception as e:
                # Calculate timing even for failed batches
                if 'batch_start_time' in locals():
                    batch_duration = time.time() - batch_start_time
                    logger.error(f"Batch {batch_num} failed after {batch_duration:.2f}s: {e}")
                else:
                    logger.error(f"Batch {batch_num} failed before timing started: {e}")

                # Create entries with empty summaries to maintain data integrity
                for entry in batch_entries:
                    processed_entry = {
                        'text': "",
                        'label': entry['label'],
                        'original_text': entry['text'],
                        'error': str(e)
                    }
                    processed_data.append(processed_entry)

                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)

                # Save partial results even for failed batches
                if output_path:
                    self._save_batch_results(processed_data[batch_start:], output_path, append=(batch_start > 0))

        # Log overall processing statistics
        total_duration = time.time() - total_start_time
        total_batches = (total_entries + batch_size - 1) // batch_size
        avg_batch_time = total_duration / total_batches if total_batches > 0 else 0
        overall_entries_per_second = total_entries / total_duration if total_duration > 0 else 0

        logger.info(f"Processing complete: {total_entries} entries in {total_duration:.2f}s "
                   f"({avg_batch_time:.2f}s/batch avg, {overall_entries_per_second:.1f} entries/sec overall)")

        return processed_data
    
    def save_summaries(self, processed_data: List[Dict[str, Any]],
                      output_path: Path) -> None:
        """Save processed summaries to a JSONL file.

        Args:
            processed_data: List of processed entries
            output_path: Path to save the output file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in processed_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')

            logger.info(f"Saved {len(processed_data)} summaries to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save summaries to {output_path}: {e}")
            raise

    def _save_batch_results(self, batch_data: List[Dict[str, Any]],
                           output_path: Path, append: bool = False) -> None:
        """Save batch results incrementally to a JSONL file.

        Args:
            batch_data: List of processed entries for this batch
            output_path: Path to save the output file
            append: Whether to append to existing file or overwrite
        """
        try:
            mode = 'a' if append else 'w'
            with open(output_path, mode, encoding='utf-8') as f:
                for entry in batch_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')

            action = "Appended" if append else "Saved"
            logger.debug(f"{action} {len(batch_data)} entries to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save batch results to {output_path}: {e}")
            # Don't raise here to allow processing to continue
            pass