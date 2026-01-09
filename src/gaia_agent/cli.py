"""
Command-line interface for GAIA Agent.
"""

import argparse
import os
import sys
from typing import Optional

from gaia_agent.agent import GAIAAgent
from gaia_agent.dataset import GAIADatasetLoader, GAIAEvaluator


def run_single_question(agent: GAIAAgent, question: str):
    """Run the agent on a single question."""
    print(f"\nQuestion: {question}")
    print("\nThinking...\n")
    answer = agent.run(question)
    print(f"\nAnswer: {answer}\n")
    return answer


def evaluate_on_dataset(
    agent: GAIAAgent,
    split: str = "validation",
    level: int = 1,
    max_questions: Optional[int] = None
):
    """Evaluate the agent on GAIA benchmark dataset."""
    loader = GAIADatasetLoader()
    evaluator = GAIAEvaluator()
    
    print(f"Loading GAIA dataset (split={split}, level={level})...")
    dataset = loader.load_dataset(split=split, level=level)
    questions = [dict(item) for item in dataset]
    
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"Loaded {len(questions)} questions\n")
    
    results = []
    for i, question_data in enumerate(questions):
        print(f"\n{'='*80}")
        print(f"Question {i+1}/{len(questions)} (ID: {question_data.get('task_id', 'N/A')})")
        print(f"{'='*80}")
        
        # Get answer from agent
        result = agent.answer_gaia_question(question_data)
        
        # Evaluate if ground truth is available
        ground_truth = question_data.get("Final answer")
        if ground_truth:
            correct = evaluator.evaluate_answer(result["answer"], ground_truth)
            result["correct"] = correct
            result["ground_truth"] = ground_truth
            print(f"\nGround Truth: {ground_truth}")
            print(f"Correct: {correct}")
        
        results.append(result)
    
    # Calculate and display accuracy
    if any("correct" in r for r in results):
        accuracy = evaluator.calculate_accuracy(results)
        print(f"\n{'='*80}")
        print(f"Overall Accuracy: {accuracy:.2%} ({sum(r.get('correct', False) for r in results)}/{len(results)})")
        print(f"{'='*80}\n")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GAIA Agent - AI assistant using Hugging Face smolagents"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="Ask a single question to the agent"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on GAIA benchmark dataset"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to use for evaluation (default: validation)"
    )
    
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="GAIA difficulty level (default: 1)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum number of questions to evaluate (for testing)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Hugging Face model ID to use"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token (or set HF_TOKEN env variable)"
    )
    
    args = parser.parse_args()
    
    # Check for HF token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: No Hugging Face API token provided. Set HF_TOKEN environment variable or use --token")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
    
    # Initialize agent
    print("Initializing GAIA Agent...")
    agent = GAIAAgent(model_id=args.model, api_token=token)
    print("Agent ready!\n")
    
    # Run based on mode
    if args.question:
        run_single_question(agent, args.question)
    elif args.evaluate:
        evaluate_on_dataset(
            agent,
            split=args.split,
            level=args.level,
            max_questions=args.max_questions
        )
    else:
        parser.print_help()
        print("\nError: Please specify either --question or --evaluate")
        sys.exit(1)


if __name__ == "__main__":
    main()
