"""Tests for the GAIA dataset module."""

import pytest
from unittest.mock import Mock, patch
from gaia_agent.dataset import GAIADatasetLoader, GAIAEvaluator


class TestGAIADatasetLoader:
    """Test cases for GAIADatasetLoader class."""
    
    @patch('gaia_agent.dataset.load_dataset')
    def test_load_dataset(self, mock_load):
        """Test loading the GAIA dataset."""
        mock_dataset = Mock()
        mock_load.return_value = mock_dataset
        
        loader = GAIADatasetLoader()
        result = loader.load_dataset(split="validation")
        
        assert result == mock_dataset
        mock_load.assert_called_once()
    
    @patch('gaia_agent.dataset.load_dataset')
    def test_load_dataset_with_level_filter(self, mock_load):
        """Test loading dataset with level filter."""
        mock_dataset = Mock()
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_load.return_value = mock_dataset
        
        loader = GAIADatasetLoader()
        result = loader.load_dataset(split="validation", level=1)
        
        mock_dataset.filter.assert_called_once()
    
    @patch('gaia_agent.dataset.load_dataset')
    def test_get_level_1_questions(self, mock_load):
        """Test getting Level 1 questions."""
        mock_item = {
            "task_id": "test-001",
            "Question": "Test question",
            "Level": 1
        }
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_item]))
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_load.return_value = mock_dataset
        
        loader = GAIADatasetLoader()
        questions = loader.get_level_1_questions(split="validation")
        
        assert len(questions) == 1
        assert questions[0]["task_id"] == "test-001"


class TestGAIAEvaluator:
    """Test cases for GAIAEvaluator class."""
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        evaluator = GAIAEvaluator()
        
        assert evaluator.normalize_answer("  Paris  ") == "paris"
        assert evaluator.normalize_answer("PARIS") == "paris"
        assert evaluator.normalize_answer("42") == "42"
        assert evaluator.normalize_answer(None) == ""
    
    def test_evaluate_answer_exact_match(self):
        """Test exact answer evaluation."""
        evaluator = GAIAEvaluator()
        
        assert evaluator.evaluate_answer("Paris", "Paris") == True
        assert evaluator.evaluate_answer("paris", "PARIS") == True
        assert evaluator.evaluate_answer("  Paris  ", "paris") == True
    
    def test_evaluate_answer_no_match(self):
        """Test answer evaluation with no match."""
        evaluator = GAIAEvaluator()
        
        assert evaluator.evaluate_answer("Paris", "London") == False
        assert evaluator.evaluate_answer("42", "43") == False
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        evaluator = GAIAEvaluator()
        
        results = [
            {"correct": True},
            {"correct": True},
            {"correct": False},
            {"correct": True},
        ]
        
        accuracy = evaluator.calculate_accuracy(results)
        assert accuracy == 0.75
    
    def test_calculate_accuracy_empty(self):
        """Test accuracy calculation with empty results."""
        evaluator = GAIAEvaluator()
        
        accuracy = evaluator.calculate_accuracy([])
        assert accuracy == 0.0
    
    def test_calculate_accuracy_all_correct(self):
        """Test accuracy calculation with all correct answers."""
        evaluator = GAIAEvaluator()
        
        results = [
            {"correct": True},
            {"correct": True},
            {"correct": True},
        ]
        
        accuracy = evaluator.calculate_accuracy(results)
        assert accuracy == 1.0
