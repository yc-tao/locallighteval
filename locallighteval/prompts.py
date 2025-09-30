"""Prompt management module for clinical summarization."""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml


class PromptManager:
    """Manages prompts for clinical summarization tasks."""

    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None):
        """Initialize the prompt manager.

        Args:
            prompt_config: Optional dictionary containing prompt configurations
        """
        self.prompt_config = prompt_config or self._get_default_prompts()

    def _get_default_prompts(self) -> Dict[str, Any]:
        """Get default prompts for clinical summarization."""
        return {
            "clinical_summary": {
                "system": "You are a clinical summarization assistant. Extract concise, accurate summaries from discharge summaries.",
                "user": "{full_note}\n\nExtract a short summary of this discharge summary between <summary> and </summary>. Be concise and to the point. Put your response in a json format, where your answer is in the key 'answer'."
            }
        }

    @classmethod
    def from_config_file(cls, config_path: Path) -> "PromptManager":
        """Create a PromptManager from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            PromptManager instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
        return cls(prompt_config)

    def get_prompt(self, prompt_type: str = "clinical_summary") -> Dict[str, str]:
        """Get system and user prompts for a specific prompt type.

        Args:
            prompt_type: Type of prompt to retrieve

        Returns:
            Dictionary containing 'system' and 'user' prompts
        """
        if prompt_type not in self.prompt_config:
            raise ValueError(f"Prompt type '{prompt_type}' not found in configuration")

        prompt_data = self.prompt_config[prompt_type]
        return {
            "system": prompt_data.get("system", ""),
            "user": prompt_data.get("user", "")
        }

    def format_prompt(self, prompt_type: str = "clinical_summary", **kwargs) -> Dict[str, str]:
        """Format prompts with provided variables.

        Args:
            prompt_type: Type of prompt to format
            **kwargs: Variables to substitute in the prompts

        Returns:
            Dictionary containing formatted 'system' and 'user' prompts
        """
        prompts = self.get_prompt(prompt_type)
        return {
            "system": prompts["system"].format(**kwargs),
            "user": prompts["user"].format(**kwargs)
        }

    def get_combined_prompt(self, prompt_type: str = "clinical_summary", **kwargs) -> str:
        """Get a combined prompt (system + user) for backward compatibility.

        Args:
            prompt_type: Type of prompt to format
            **kwargs: Variables to substitute in the prompts

        Returns:
            Combined prompt string
        """
        formatted = self.format_prompt(prompt_type, **kwargs)
        if formatted["system"]:
            return f"System: {formatted['system']}\n\nUser: {formatted['user']}"
        return formatted["user"]