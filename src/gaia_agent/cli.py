"""
Command-line interface for GAIA Agent.

Provides commands for:
- Asking single questions
- Evaluating on GAIA benchmark dataset
- Saving questions to JSON files
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from gaia_agent.agent import GAIAAgent
from gaia_agent.dataset import GAIADatasetLoader, GAIAEvaluator


# Configuration
DEFAULT_DELAY_SECONDS = 2  # Delay between questions to avoid rate limiting


def save_questions_to_json(
    output_file: str,
    split: str = "validation",
    level: int = 1,
    max_questions: Optional[int] = None,
):
    """Save GAIA questions to a JSON file."""
    loader = GAIADatasetLoader()

    print(f"Loading GAIA dataset (split={split}, level={level})...")
    dataset = loader.load_dataset(split=split, level=level)
    questions = [dict(item) for item in dataset]

    if max_questions:
        questions = questions[:max_questions]

    # Convert any non-serializable fields
    for q in questions:
        # Handle file_path bytes if present
        if "file_path" in q and isinstance(q["file_path"], bytes):
            q["file_path"] = q["file_path"].decode("utf-8", errors="replace")

    print(f"Saving {len(questions)} questions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False, default=str)

    print(
        f"Successfully saved {len(questions)} level {level} questions to {output_file}"
    )
    return questions


def run_single_question(agent: GAIAAgent, question: str):
    """Run the agent on a single question."""
    print(f"\nQuestion: {question}")
    print("\n" + "=" * 60)
    print("Agent is thinking...")
    print("=" * 60 + "\n")

    answer = agent.run(question)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"\nFull Response:\n{answer}")

    # Try to extract FINAL_ANSWER if present
    extracted = agent._extract_final_answer(answer)
    if extracted != answer:
        print(f"\nExtracted Answer: {extracted}")

    return answer


def save_detailed_results(
    results: List[Dict[str, Any]], output_file: str, evaluator: GAIAEvaluator
):
    """
    Save detailed evaluation results to a JSON file.

    Args:
        results: List of result dictionaries
        output_file: Path to output JSON file
        evaluator: Evaluator instance for normalization info
    """
    # Add normalized versions for debugging
    for result in results:
        if "answer" in result:
            result["normalized_answer"] = evaluator.normalize_answer(result["answer"])
        if "ground_truth" in result:
            result["normalized_ground_truth"] = evaluator.normalize_answer(
                result["ground_truth"]
            )

    # Add summary statistics
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    with_files = sum(1 for r in results if r.get("has_file", False))
    correct_with_files = sum(
        1 for r in results if r.get("correct", False) and r.get("has_file", False)
    )
    correct_without_files = sum(
        1 for r in results if r.get("correct", False) and not r.get("has_file", False)
    )

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "questions_with_files": with_files,
            "correct_with_files": correct_with_files,
            "correct_without_files": correct_without_files,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDetailed results saved to: {output_file}")


def evaluate_on_dataset(
    agent: GAIAAgent,
    split: str = "validation",
    level: int = 1,
    max_questions: Optional[int] = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    output_file: Optional[str] = None,
    max_retries: int = 2,
    retry_delay: float = 2.0,
    nofile: bool = False,
):
    """
    Evaluate the agent on GAIA benchmark dataset.

    Args:
        agent: The GAIAAgent instance
        split: Dataset split ("validation" or "test")
        level: GAIA difficulty level (1, 2, or 3)
        max_questions: Maximum questions to evaluate (None for all)
        delay_seconds: Delay between questions to avoid rate limiting
        output_file: Optional path to save detailed results JSON
        max_retries: Maximum retry attempts for transient API failures
        retry_delay: Base delay in seconds between retries
        nofile: If True, filter out questions that have attached files
    """
    loader = GAIADatasetLoader()
    evaluator = GAIAEvaluator()

    print(f"Loading GAIA dataset (split={split}, level={level})...")
    dataset = loader.load_dataset(split=split, level=level)
    questions = [dict(item) for item in dataset]

    # Filter out questions with files if --nofile is specified
    if nofile:
        original_count = len(questions)
        questions = [q for q in questions if not q.get("file_name")]
        print(f"Filtered to {len(questions)} questions without files (removed {original_count - len(questions)} with files)")

    if max_questions:
        questions = questions[:max_questions]

    total_questions = len(questions)
    print(f"Loaded {total_questions} questions")
    print(f"Delay between questions: {delay_seconds}s")
    print("\n" + "=" * 80)

    results = []
    correct_count = 0

    # Create progress bar if tqdm is available
    if TQDM_AVAILABLE:
        question_iterator = tqdm(
            enumerate(questions), total=total_questions, desc="Evaluating"
        )
    else:
        question_iterator = enumerate(questions)
        print("(Install tqdm for progress bar: pip install tqdm)\n")

    for i, question_data in question_iterator:
        task_id = question_data.get("task_id", "N/A")
        question_text = question_data.get("Question", "")
        has_file = bool(question_data.get("file_name"))

        # Print question header
        print(f"\n{'=' * 80}")
        print(f"Question {i + 1}/{total_questions}")
        print(f"Task ID: {task_id}")
        print(
            f"Has File: {'Yes - ' + question_data.get('file_name', '') if has_file else 'No'}"
        )
        print(f"{'=' * 80}")

        # Print truncated question
        max_display_len = 300
        if len(question_text) > max_display_len:
            print(f"\nQuestion: {question_text[:max_display_len]}...")
        else:
            print(f"\nQuestion: {question_text}")

        print("\n" + "-" * 40)
        print("Agent working...")
        print("-" * 40)

        # Get answer from agent
        start_time = time.time()
        result = agent.answer_gaia_question(
            question_data,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        elapsed_time = time.time() - start_time
        result["elapsed_time"] = elapsed_time

        print(f"\nTime taken: {elapsed_time:.1f}s")

        # Print the agent's extracted answer
        print(f"\n{'=' * 40}")
        print("ANSWER")
        print(f"{'=' * 40}")
        print(f"Agent's Answer: {result['answer']}")

        # Evaluate if ground truth is available
        ground_truth = question_data.get("Final answer")
        if ground_truth:
            correct = evaluator.evaluate_answer(result["answer"], ground_truth)
            result["correct"] = correct
            result["ground_truth"] = ground_truth

            if correct:
                correct_count += 1

            print(f"Ground Truth:   {ground_truth}")
            print(f"\nResult: {'[OK] CORRECT' if correct else '[X] INCORRECT'}")

            # Show normalized versions if incorrect (for debugging)
            if not correct:
                norm_answer = evaluator.normalize_answer(result["answer"])
                norm_truth = evaluator.normalize_answer(ground_truth)
                print(f"\n[Debug] Normalized Answer: '{norm_answer}'")
                print(f"[Debug] Normalized Truth:  '{norm_truth}'")

        # Print raw answer (truncated) for debugging
        raw_answer = result.get("raw_answer", "")
        if raw_answer and raw_answer != result["answer"]:
            print(f"\n{'=' * 40}")
            print("RAW OUTPUT (truncated)")
            print(f"{'=' * 40}")
            max_raw_len = 500
            if len(raw_answer) > max_raw_len:
                print(f"{raw_answer[:max_raw_len]}...")
            else:
                print(raw_answer)

        results.append(result)

        # Running accuracy
        print(
            f"\n[Running Accuracy: {correct_count}/{i + 1} = {correct_count / (i + 1) * 100:.1f}%]"
        )

        # Delay before next question (unless it's the last one)
        if i < total_questions - 1 and delay_seconds > 0:
            if not TQDM_AVAILABLE:
                print(f"\nWaiting {delay_seconds}s before next question...")
            time.sleep(delay_seconds)

    # Calculate and display final accuracy
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if any("correct" in r for r in results):
        accuracy = evaluator.calculate_accuracy(results)

        # Breakdown by file presence
        results_with_files = [r for r in results if r.get("has_file", False)]
        results_without_files = [r for r in results if not r.get("has_file", False)]

        print(f"\nOverall Accuracy: {accuracy:.1%} ({correct_count}/{total_questions})")
        print(f"\nBreakdown:")

        if results_without_files:
            acc_no_file = evaluator.calculate_accuracy(results_without_files)
            correct_no_file = sum(
                1 for r in results_without_files if r.get("correct", False)
            )
            print(
                f"  - Questions without files: {acc_no_file:.1%} ({correct_no_file}/{len(results_without_files)})"
            )

        if results_with_files:
            acc_with_file = evaluator.calculate_accuracy(results_with_files)
            correct_with_file = sum(
                1 for r in results_with_files if r.get("correct", False)
            )
            print(
                f"  - Questions with files:    {acc_with_file:.1%} ({correct_with_file}/{len(results_with_files)})"
            )

        # Average time
        avg_time = sum(r.get("elapsed_time", 0) for r in results) / len(results)
        print(f"\nAverage time per question: {avg_time:.1f}s")
        print(
            f"Total evaluation time: {sum(r.get('elapsed_time', 0) for r in results):.1f}s"
        )

    print("=" * 80 + "\n")

    # Save detailed results if output file specified
    if output_file:
        save_detailed_results(results, output_file, evaluator)
    else:
        # Default output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"gaia_results_{timestamp}.json"
        save_detailed_results(results, default_output, evaluator)

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GAIA Agent - AI assistant using Hugging Face smolagents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  gaia-agent --question "What is the capital of France?"
  
  # Evaluate on 5 questions (for testing)
  gaia-agent --evaluate --level 1 --max-questions 5
  
  # Full evaluation with custom output file
  gaia-agent --evaluate --level 1 --output results.json
  
  # Use a different model
  gaia-agent --question "Hello" --model "meta-llama/Llama-3.1-70B-Instruct"
""",
    )

    parser.add_argument(
        "--question", type=str, help="Ask a single question to the agent"
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on GAIA benchmark dataset"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to use for evaluation (default: validation)",
    )

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="GAIA difficulty level (default: 1)",
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum number of questions to evaluate (for testing)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Delay in seconds between questions (default: {DEFAULT_DELAY_SECONDS})",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts for transient API failures (default: 2)",
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds between retries with exponential backoff (default: 2.0)",
    )

    parser.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        help="Output file for detailed results JSON (default: auto-generated)",
    )

    parser.add_argument(
        "--nofile",
        action="store_true",
        help="Filter out questions that have attached files (evaluate only text-based questions)",
    )

    parser.add_argument(
        "--save-questions",
        type=str,
        metavar="FILE",
        help="Save questions to a JSON file (use with --level and --split)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Azure OpenAI deployment name (default: gpt-5-mini)",
    )

    parser.add_argument(
        "--token",
        type=str,
        help="(Deprecated) Previously used for Hugging Face token, now ignored",
    )

    args = parser.parse_args()

    # Handle save-questions (doesn't need agent)
    if args.save_questions:
        save_questions_to_json(
            output_file=args.save_questions,
            split=args.split,
            level=args.level,
            max_questions=args.max_questions,
        )
        return

    # Check for Azure OpenAI credentials
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not azure_endpoint or not azure_key:
        print("Error: Azure OpenAI credentials not configured.")
        print("Set the following environment variables:")
        print("  AZURE_OPENAI_ENDPOINT - Your Azure OpenAI endpoint URL")
        print("  AZURE_OPENAI_API_KEY - Your Azure OpenAI API key")
        print(
            "  AZURE_OPENAI_API_VERSION - (optional) API version, defaults to 2024-10-01-preview"
        )
        sys.exit(1)

    # Create log file for execution details
    log_file = None
    log_file_path = None
    if args.evaluate:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"gaia_logs_{timestamp}.txt"
        log_file = open(log_file_path, "w", encoding="utf-8")
        log_file.write(f"GAIA Agent Execution Log\n")
        log_file.write(f"Generated: {datetime.now().isoformat()}\n")
        log_file.write(f"Model: {args.model or 'gpt-5-mini'}\n")
        log_file.write(f"Level: {args.level}\n")
        log_file.write(f"Max Questions: {args.max_questions}\n")
        log_file.write(f"{'='*60}\n\n")
        print(f"Execution logs will be saved to: {log_file_path}")

    try:
        # Initialize agent
        print("=" * 60)
        print("GAIA Agent Initialization")
        print("=" * 60)
        model_name = args.model or "gpt-5-mini"
        print(f"Model: {model_name}")
        print("Initializing...")

        agent = GAIAAgent(model_id=args.model, log_file=log_file)
        print("Agent ready!\n")

        # Run based on mode
        if args.question:
            run_single_question(agent, args.question)
        elif args.evaluate:
            evaluate_on_dataset(
                agent,
                split=args.split,
                level=args.level,
                max_questions=args.max_questions,
                delay_seconds=args.delay,
                output_file=args.output,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                nofile=args.nofile,
            )
        else:
            parser.print_help()
            print(
                "\nError: Please specify either --question, --evaluate, or --save-questions"
            )
            sys.exit(1)
    finally:
        if log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Log completed: {datetime.now().isoformat()}\n")
            log_file.close()
            print(f"\nExecution logs saved to: {log_file_path}")


if __name__ == "__main__":
    main()
