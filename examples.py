#!/usr/bin/env python3
"""
Example script demonstrating how to use the GAIA Agent.
"""

import os
from gaia_agent.agent import GAIAAgent
from gaia_agent.dataset import GAIADatasetLoader


def example_single_question():
    """Example: Ask a single question."""
    print("="*80)
    print("Example 1: Single Question")
    print("="*80 + "\n")
    
    # Initialize agent
    agent = GAIAAgent()
    
    # Ask a question
    question = "What is the capital of France?"
    print(f"Question: {question}")
    answer = agent.run(question)
    print(f"Answer: {answer}\n")


def example_gaia_dataset():
    """Example: Load and answer GAIA dataset questions."""
    print("="*80)
    print("Example 2: GAIA Dataset")
    print("="*80 + "\n")
    
    # Load dataset
    loader = GAIADatasetLoader()
    questions = loader.get_level_1_questions(split="validation")
    
    print(f"Loaded {len(questions)} Level 1 questions from GAIA validation set\n")
    
    # Show first question
    if questions:
        first_q = questions[0]
        print("First question:")
        print(f"  Task ID: {first_q.get('task_id', 'N/A')}")
        print(f"  Question: {first_q.get('Question', 'N/A')}")
        print(f"  Level: {first_q.get('Level', 'N/A')}")
        
        # Initialize agent and answer
        print("\nGetting answer from agent...")
        agent = GAIAAgent()
        result = agent.answer_gaia_question(first_q)
        print(f"Answer: {result['answer']}\n")


def main():
    """Run examples."""
    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not set.")
        print("Some features may not work without a Hugging Face API token.")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
    
    try:
        # Run single question example
        example_single_question()
        
        # Run GAIA dataset example
        # Uncomment the line below to run (requires dataset download)
        # example_gaia_dataset()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed all requirements:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
