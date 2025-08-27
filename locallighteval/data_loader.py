"""Data loading utilities for LocalLightEval."""

import json
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Tuple
from loguru import logger
from tqdm import tqdm


class EvalDataset:
    """Dataset class for evaluation data."""
    
    def __init__(
        self,
        data_path: str,
        text_key: str = "text",
        label_key: str = "label",
        max_samples: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.text_key = text_key
        self.label_key = label_key
        self.max_samples = max_samples
        self.data_format = None  # Will be detected: 'jsonl' or 'json_array'
        self._data_cache = None  # For JSON array format
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self._detect_format()
        self._validate_data()
        
    def _detect_format(self) -> None:
        """Detect whether the file is JSON Lines or JSON Array format."""
        logger.info(f"Detecting format of data file: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # Read first non-empty line
            first_line = ""
            for line in f:
                stripped = line.strip()
                if stripped:
                    first_line = stripped
                    break
            
            if first_line.startswith('['):
                self.data_format = 'json_array'
                logger.info("Detected JSON array format")
            elif first_line.startswith('{'):
                self.data_format = 'jsonl'
                logger.info("Detected JSON Lines format")
            else:
                raise ValueError(f"Unable to detect data format. First line: {first_line[:100]}...")
    
    def _load_json_array(self) -> List[Dict[str, Any]]:
        """Load data from JSON array format."""
        if self._data_cache is not None:
            return self._data_cache
            
        logger.info("Loading JSON array data...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected JSON array, got {type(data)}")
                self._data_cache = data
                logger.info(f"Loaded {len(data)} items from JSON array")
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON array format: {e}")
    
    def _validate_data(self) -> None:
        """Validate the data format and structure."""
        logger.info(f"Validating data file: {self.data_path}")
        
        sample_count = 0
        valid_count = 0
        
        if self.data_format == 'json_array':
            # Validate JSON array data
            data = self._load_json_array()
            validation_samples = min(100, len(data))
            
            for idx, item in enumerate(data[:validation_samples]):
                sample_count += 1
                
                if not isinstance(item, dict):
                    logger.warning(f"Item {idx}: Expected dict, got {type(item)}")
                    continue
                
                if self.text_key not in item:
                    logger.warning(f"Item {idx}: Missing text key '{self.text_key}'")
                    continue
                
                if self.label_key not in item:
                    logger.warning(f"Item {idx}: Missing label key '{self.label_key}'")
                    continue
                
                text = item[self.text_key]
                label = item[self.label_key]
                
                if not isinstance(text, str):
                    logger.warning(f"Item {idx}: Text should be string, got {type(text)}")
                    continue
                
                if not isinstance(label, (int, bool)):
                    logger.warning(f"Item {idx}: Label should be int or bool, got {type(label)}")
                    continue
                
                if isinstance(label, bool):
                    pass  # Boolean labels are fine
                elif isinstance(label, int) and label not in [0, 1]:
                    logger.warning(f"Item {idx}: Integer label should be 0 or 1, got {label}")
                    continue
                
                valid_count += 1
                
        else:
            # Validate JSON Lines data
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                        
                    sample_count += 1
                    if sample_count > 100:  # Only validate first 100 samples
                        break
                    
                    try:
                        data = json.loads(line)
                        
                        if not isinstance(data, dict):
                            logger.warning(f"Line {line_num}: Expected dict, got {type(data)}")
                            continue
                        
                        if self.text_key not in data:
                            logger.warning(f"Line {line_num}: Missing text key '{self.text_key}'")
                            continue
                        
                        if self.label_key not in data:
                            logger.warning(f"Line {line_num}: Missing label key '{self.label_key}'")
                            continue
                        
                        text = data[self.text_key]
                        label = data[self.label_key]
                        
                        if not isinstance(text, str):
                            logger.warning(f"Line {line_num}: Text should be string, got {type(text)}")
                            continue
                        
                        if not isinstance(label, (int, bool)):
                            logger.warning(f"Line {line_num}: Label should be int or bool, got {type(label)}")
                            continue
                        
                        if isinstance(label, bool):
                            pass  # Boolean labels are fine
                        elif isinstance(label, int) and label not in [0, 1]:
                            logger.warning(f"Line {line_num}: Integer label should be 0 or 1, got {label}")
                            continue
                        
                        valid_count += 1
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue
        
        if valid_count == 0:
            raise ValueError("No valid samples found in the data file")
        
        logger.info(f"Validation complete. Found {valid_count}/{sample_count} valid samples")
    
    def __len__(self) -> int:
        """Get the total number of samples."""
        if self.data_format == 'json_array':
            data = self._load_json_array()
            total_count = len(data)
            return min(total_count, self.max_samples) if self.max_samples else total_count
        else:
            # JSON Lines format
            count = 0
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
                        if self.max_samples and count >= self.max_samples:
                            break
            return count
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset."""
        count = 0
        
        if self.data_format == 'json_array':
            # Handle JSON array format
            data = self._load_json_array()
            
            for idx, item in enumerate(data):
                if self.max_samples and count >= self.max_samples:
                    break
                
                if not isinstance(item, dict):
                    logger.warning(f"Item {idx}: Skipping non-dict entry")
                    continue
                
                if self.text_key not in item or self.label_key not in item:
                    logger.warning(f"Item {idx}: Skipping entry with missing keys")
                    continue
                
                text = item[self.text_key]
                label = item[self.label_key]
                
                if not isinstance(text, str):
                    logger.warning(f"Item {idx}: Skipping entry with non-string text")
                    continue
                
                # Convert label to int
                if isinstance(label, bool):
                    label = int(label)
                elif isinstance(label, int):
                    if label not in [0, 1]:
                        logger.warning(f"Item {idx}: Skipping entry with invalid label {label}")
                        continue
                else:
                    logger.warning(f"Item {idx}: Skipping entry with invalid label type {type(label)}")
                    continue
                
                yield {
                    "text": text,
                    "label": label,
                    "line_num": idx + 1,  # Use index as line number
                    "original": item
                }
                
                count += 1
        
        else:
            # Handle JSON Lines format
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    if self.max_samples and count >= self.max_samples:
                        break
                    
                    try:
                        data = json.loads(line)
                        
                        if not isinstance(data, dict):
                            logger.warning(f"Line {line_num}: Skipping non-dict entry")
                            continue
                        
                        if self.text_key not in data or self.label_key not in data:
                            logger.warning(f"Line {line_num}: Skipping entry with missing keys")
                            continue
                        
                        text = data[self.text_key]
                        label = data[self.label_key]
                        
                        if not isinstance(text, str):
                            logger.warning(f"Line {line_num}: Skipping entry with non-string text")
                            continue
                        
                        # Convert label to int
                        if isinstance(label, bool):
                            label = int(label)
                        elif isinstance(label, int):
                            if label not in [0, 1]:
                                logger.warning(f"Line {line_num}: Skipping entry with invalid label {label}")
                                continue
                        else:
                            logger.warning(f"Line {line_num}: Skipping entry with invalid label type {type(label)}")
                            continue
                        
                        yield {
                            "text": text,
                            "label": label,
                            "line_num": line_num,
                            "original": data
                        }
                        
                        count += 1
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Skipping invalid JSON - {e}")
                        continue
    
    def get_batches(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Get data in batches."""
        batch = []
        
        for item in tqdm(self, desc="Loading data", total=len(self)):
            batch.append(item)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining items
            yield batch
    
    def get_texts_and_labels(self) -> Tuple[List[str], List[int]]:
        """Get all texts and labels as separate lists."""
        texts = []
        labels = []
        
        for item in tqdm(self, desc="Loading all data", total=len(self)):
            texts.append(item["text"])
            labels.append(item["label"])
        
        return texts, labels


def load_dataset(
    data_path: str,
    text_key: str = "text",
    label_key: str = "label",
    max_samples: Optional[int] = None
) -> EvalDataset:
    """Load and return an evaluation dataset."""
    logger.info(f"Loading dataset from {data_path}")
    
    dataset = EvalDataset(
        data_path=data_path,
        text_key=text_key,
        label_key=label_key,
        max_samples=max_samples
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    return dataset