"""Text summarization module for generating clinical summaries."""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from rich.progress import Progress, TaskID

from .inference import VLLMInferenceEngine
from .prompts import PromptManager


class ClinicalSummarizationEngine:
    """Engine for generating clinical summaries from discharge summaries."""
    
    def __init__(self, inference_engine: VLLMInferenceEngine,
                 prompt_manager: Optional[PromptManager] = None,
                 prompt_type: str = "clinical_summary",
                 debug: bool = False,
                 cleanup_discharge_text: bool = False):
        """Initialize the summarization engine.

        Args:
            inference_engine: The vLLM inference engine to use for generation
            prompt_manager: Optional prompt manager for handling prompts
            prompt_type: Type of prompt to use from the prompt manager
            debug: If True, print raw LLM responses for debugging
            cleanup_discharge_text: If True, clean up discharge text by removing extra headers/footers
        """
        self.inference_engine = inference_engine
        self.prompt_manager = prompt_manager or PromptManager()
        self.prompt_type = prompt_type
        self.debug = debug
        self.cleanup_discharge_text = cleanup_discharge_text
    
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

    def _cleanup_discharge_text(self, text: str) -> str:
        """Clean up discharge summary text by removing extra headers and footers.

        Args:
            text: The discharge summary text to clean up

        Returns:
            Cleaned discharge summary text
        """
        cleaned_text = text

        # Remove the prefix pattern: "\n\n        Here is the discharge summary:\n\n"
        # This pattern may have variations in whitespace
        prefix_patterns = [
            r'^\s*Here is the discharge summary:\s*',
            r'^\s*here is the discharge summary:\s*',
        ]

        for pattern in prefix_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Remove trailing "Summary:" at the end
        # Also handle variations with whitespace
        suffix_patterns = [
            r'\s*Summary:\s*$',
            r'\s*summary:\s*$',
        ]

        for pattern in suffix_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Clean up any excessive whitespace at the beginning or end
        cleaned_text = cleaned_text.strip()

        return cleaned_text

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

        # Clean up discharge text if enabled
        processed_notes = full_notes
        if self.cleanup_discharge_text:
            processed_notes = [self._cleanup_discharge_text(note) for note in full_notes]
            logger.debug(f"Applied discharge text cleanup to {len(full_notes)} notes")

        # Format the prompts using the prompt manager - always use dict format
        prompts = []
        for i, note in enumerate(processed_notes):
            formatted_prompts = self.prompt_manager.format_prompt(self.prompt_type, full_note=note)
            prompts.append(formatted_prompts)

            # Print input prompt in debug mode
            if self.debug:
                logger.debug(f"Input prompt to LLM (batch item {i+1}): {formatted_prompts}")

        # Generate responses using the inference engine batch processing
        try:
            responses = self.inference_engine.generate(prompts)
            if self.debug:
                logger.debug(f"Generated text: {responses}")
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

            # Extract text between <summary> and </summary> tags from raw output
            summary_match = re.search(r'<summary>(.*?)</summary>', generated_text, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1).strip()
                # If extracted summary is too short (< 30 chars), fall back to original text
                if len(summary) < 30:
                    logger.warning(f"Extracted summary for item {i+1} is too short ({len(summary)} chars), using original text")
                    summaries.append(full_notes[i])
                else:
                    logger.debug(f"Extracted summary for item {i+1}: {summary[:100]}...")
                    summaries.append(summary)
            else:
                # Try to extract JSON format with "answer" field, use longest if multiple found
                json_matches = re.findall(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', generated_text, re.DOTALL)
                if json_matches:
                    summary = max(json_matches, key=len).strip()
                    if len(summary) < 30:
                        logger.warning(f"Extracted JSON summary for item {i+1} is too short ({len(summary)} chars), using original text")
                        summaries.append(full_notes[i])
                    else:
                        logger.debug(f"Extracted JSON summary for item {i+1}: {summary[:100]}...")
                        summaries.append(summary)
                else:
                    # If no tags or JSON found, return the raw output as is (but log warning)
                    logger.warning(f"Generated text for item {i+1} does not contain <summary> tags or JSON format, using raw output")
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
