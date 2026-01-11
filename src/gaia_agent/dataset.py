"""
GAIA Dataset Loader and Evaluator

This module provides utilities to load and evaluate on the GAIA benchmark dataset.
"""

from typing import List, Dict, Any, Optional
from datasets import load_dataset


class GAIADatasetLoader:
    """Load and manage GAIA benchmark dataset."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the GAIA dataset loader.

        Args:
            cache_dir: Directory to cache the dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None

    def load_dataset(self, split: str = "validation", level: Optional[int] = None):
        """
        Load the GAIA dataset.

        Args:
            split: Dataset split to load ("train", "validation", "test")
            level: Filter by difficulty level (1, 2, or 3). None for all levels.

        Returns:
            The loaded dataset
        """
        # Load the GAIA benchmark dataset
        self.dataset = load_dataset(
            "gaia-benchmark/GAIA", "2023_all", split=split, cache_dir=self.cache_dir
        )

        # Filter by level if specified (Level field is a string in the dataset)
        if level is not None:
            level_str = str(level)
            self.dataset = self.dataset.filter(lambda x: x["Level"] == level_str)

        return self.dataset

    def get_level_1_questions(self, split: str = "validation") -> List[Dict[str, Any]]:
        """
        Get all Level 1 questions from the dataset.

        Args:
            split: Dataset split to load

        Returns:
            List of Level 1 questions as dictionaries
        """
        dataset = self.load_dataset(split=split, level=1)
        return [dict(item) for item in dataset]

    def get_question_by_id(
        self, task_id: str, split: str = "validation"
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific question by its task_id.

        Args:
            task_id: The task_id to search for
            split: Dataset split to search in

        Returns:
            Question data as dictionary, or None if not found
        """
        if self.dataset is None:
            self.load_dataset(split=split)

        filtered = self.dataset.filter(lambda x: x["task_id"] == task_id)
        if len(filtered) > 0:
            return dict(filtered[0])
        return None


class GAIAEvaluator:
    """Evaluate agent performance on GAIA benchmark."""

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        Normalize an answer for comparison with GAIA ground truth.

        Applies various normalization techniques to improve matching:
        - Case normalization
        - Whitespace normalization
        - Punctuation cleanup
        - Number format normalization
        - Common variation handling

        Args:
            answer: The answer to normalize

        Returns:
            Normalized answer string
        """
        import re

        if answer is None:
            return ""

        # Convert to string and strip whitespace
        answer = str(answer).strip()

        # Case insensitive comparison
        answer = answer.lower()

        # Remove trailing/leading punctuation
        answer = answer.strip(".,;!?:'\"")

        # Normalize whitespace (multiple spaces to single)
        answer = " ".join(answer.split())

        # Remove thousands separators in numbers (1,000 -> 1000)
        answer = re.sub(r"(\d),(\d)", r"\1\2", answer)

        # Normalize common variations
        answer = answer.replace(" percent", "%")
        answer = answer.replace("percent", "%")

        # Remove common prefixes/suffixes that don't affect meaning
        prefixes_to_remove = ["the ", "a ", "an "]
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix) :]
                break

        # Normalize quotes
        answer = answer.replace('"', "").replace("'", "").replace("`", "")

        # Remove trailing period if present
        answer = answer.rstrip(".")

        return answer.strip()

    @staticmethod
    def evaluate_answer(predicted: str, ground_truth: str) -> bool:
        """
        Evaluate if a predicted answer matches the ground truth.

        Args:
            predicted: The predicted answer
            ground_truth: The ground truth answer

        Returns:
            True if answers match, False otherwise
        """
        pred_normalized = GAIAEvaluator.normalize_answer(predicted)
        gt_normalized = GAIAEvaluator.normalize_answer(ground_truth)

        # Simple exact match after normalization
        return pred_normalized == gt_normalized

    @staticmethod
    def calculate_accuracy(results: List[Dict[str, Any]]) -> float:
        """
        Calculate accuracy from a list of results.

        Args:
            results: List of result dictionaries with 'correct' key

        Returns:
            Accuracy as a float between 0 and 1
        """
        if not results:
            return 0.0

        correct = sum(1 for r in results if r.get("correct", False))
        return correct / len(results)
