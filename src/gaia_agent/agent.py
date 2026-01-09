"""
GAIA Agent: A general AI assistant agent using Hugging Face smolagents.

This module provides an AI agent capable of answering GAIA benchmark Level 1 questions.
"""

import os
from typing import Optional, Dict, Any
from smolagents import CodeAgent, ApiModel, DuckDuckGoSearchTool, VisitWebpageTool


class GAIAAgent:
    """
    A general AI assistant agent built with Hugging Face smolagents.
    
    This agent is designed to answer questions from the GAIA benchmark,
    particularly Level 1 questions that require basic reasoning, web search,
    and information retrieval.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        api_token: Optional[str] = None,
        additional_tools: Optional[list] = None
    ):
        """
        Initialize the GAIA agent.
        
        Args:
            model_id: Hugging Face model ID to use (default: uses ApiModel default)
            api_token: Hugging Face API token (default: uses HF_TOKEN env variable)
            additional_tools: Additional tools to add to the agent
        """
        # Set up API token
        if api_token:
            os.environ["HF_TOKEN"] = api_token
        
        # Initialize the model
        if model_id:
            self.model = ApiModel(model_id=model_id)
        else:
            self.model = ApiModel()
        
        # Set up default tools for GAIA benchmark
        self.tools = [
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
        ]
        
        # Add any additional tools
        if additional_tools:
            self.tools.extend(additional_tools)
        
        # Create the agent
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            additional_authorized_imports=['requests', 'bs4', 'json', 're']
        )
    
    def run(self, question: str, max_steps: int = 10) -> str:
        """
        Run the agent to answer a question.
        
        Args:
            question: The question to answer
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            The agent's answer as a string
        """
        try:
            result = self.agent.run(question, max_steps=max_steps)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def answer_gaia_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a GAIA benchmark question.
        
        Args:
            question_data: Dictionary containing GAIA question data with fields:
                - Question: The question text
                - task_id: Unique identifier
                - Level: Difficulty level
                - file_name: Optional file attachment name
                
        Returns:
            Dictionary with the answer and metadata
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")
        
        # If there's a file attachment, note it in the question
        file_name = question_data.get("file_name")
        if file_name:
            question = f"[Note: This question references a file: {file_name}]\n{question}"
        
        # Get the answer from the agent
        answer = self.run(question)
        
        return {
            "task_id": task_id,
            "question": question_data.get("Question", ""),
            "answer": answer,
            "level": question_data.get("Level", 1)
        }
