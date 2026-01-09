#!/usr/bin/env python3
"""
Demo script to verify the GAIA agent project structure and basic functionality.
This script tests core functionality without requiring an API token.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from gaia_agent import __version__
        from gaia_agent.agent import GAIAAgent
        from gaia_agent.dataset import GAIADatasetLoader, GAIAEvaluator
        from gaia_agent.cli import main
        print(f"✓ All imports successful (version: {__version__})")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_loader():
    """Test the dataset loader functionality."""
    print("\nTesting dataset loader...")
    try:
        from gaia_agent.dataset import GAIADatasetLoader
        
        loader = GAIADatasetLoader()
        print("✓ GAIADatasetLoader initialized")
        
        # Test that methods exist
        assert hasattr(loader, 'load_dataset')
        assert hasattr(loader, 'get_level_1_questions')
        assert hasattr(loader, 'get_question_by_id')
        print("✓ All dataset loader methods available")
        return True
    except Exception as e:
        print(f"✗ Dataset loader test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluator():
    """Test the evaluator functionality."""
    print("\nTesting evaluator...")
    try:
        from gaia_agent.dataset import GAIAEvaluator
        
        evaluator = GAIAEvaluator()
        
        # Test normalization
        assert evaluator.normalize_answer("  Paris  ") == "paris"
        assert evaluator.normalize_answer("PARIS") == "paris"
        print("✓ Answer normalization works")
        
        # Test evaluation
        assert evaluator.evaluate_answer("Paris", "paris") == True
        assert evaluator.evaluate_answer("Paris", "London") == False
        print("✓ Answer evaluation works")
        
        # Test accuracy calculation
        results = [
            {"correct": True},
            {"correct": False},
            {"correct": True}
        ]
        accuracy = evaluator.calculate_accuracy(results)
        assert abs(accuracy - 0.6667) < 0.01
        print("✓ Accuracy calculation works")
        
        return True
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_structure():
    """Test that CLI is properly structured."""
    print("\nTesting CLI structure...")
    try:
        from gaia_agent.cli import main
        import argparse
        
        # Check that the main function exists and is callable
        assert callable(main)
        print("✓ CLI main function available")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("GAIA Agent Project Verification")
    print("="*80)
    
    tests = [
        test_imports,
        test_dataset_loader,
        test_evaluator,
        test_cli_structure,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*80)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*80)
    
    if all(results):
        print("\n✓ All verification tests passed!")
        print("\nProject structure is ready. To use the agent:")
        print("  1. Set HF_TOKEN environment variable with your Hugging Face token")
        print("  2. Install ddgs package: pip install ddgs")
        print("  3. Run: gaia-agent --question 'Your question here'")
        print("  4. Or evaluate: gaia-agent --evaluate --level 1 --max-questions 5")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
