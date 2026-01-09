"""Tests for the GAIA agent module."""

import pytest
from unittest.mock import Mock, patch
from gaia_agent.agent import GAIAAgent


class TestGAIAAgent:
    """Test cases for GAIAAgent class."""
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        with patch('gaia_agent.agent.HfApiModel') as mock_model:
            with patch('gaia_agent.agent.CodeAgent') as mock_agent:
                agent = GAIAAgent()
                assert agent is not None
                assert hasattr(agent, 'agent')
                assert hasattr(agent, 'tools')
                assert hasattr(agent, 'model')
    
    def test_agent_with_custom_model(self):
        """Test agent initialization with custom model."""
        with patch('gaia_agent.agent.HfApiModel') as mock_model:
            with patch('gaia_agent.agent.CodeAgent') as mock_agent:
                agent = GAIAAgent(model_id="test-model")
                mock_model.assert_called_once()
    
    def test_answer_gaia_question(self):
        """Test answering a GAIA question."""
        with patch('gaia_agent.agent.HfApiModel'):
            with patch('gaia_agent.agent.CodeAgent') as mock_agent:
                agent = GAIAAgent()
                
                # Mock the agent's run method
                agent.agent.run = Mock(return_value="Test answer")
                
                question_data = {
                    "task_id": "test-001",
                    "Question": "What is 2+2?",
                    "Level": 1
                }
                
                result = agent.answer_gaia_question(question_data)
                
                assert result["task_id"] == "test-001"
                assert result["answer"] == "Test answer"
                assert result["level"] == 1
                assert result["question"] == "What is 2+2?"
    
    def test_answer_gaia_question_with_file(self):
        """Test answering a GAIA question with file attachment."""
        with patch('gaia_agent.agent.HfApiModel'):
            with patch('gaia_agent.agent.CodeAgent') as mock_agent:
                agent = GAIAAgent()
                agent.agent.run = Mock(return_value="Test answer")
                
                question_data = {
                    "task_id": "test-002",
                    "Question": "What is in the document?",
                    "Level": 1,
                    "file_name": "test.pdf"
                }
                
                result = agent.answer_gaia_question(question_data)
                
                assert result["task_id"] == "test-002"
                assert "file" in result["question"].lower() or "Note" in result["question"]
