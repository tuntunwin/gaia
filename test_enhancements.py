#!/usr/bin/env python
"""Quick test of the new enhancement functions."""

from gaia_agent.agent import (
    extract_answer_requirements,
    validate_answer_format,
    normalize_answer,
    preprocess_question,
)

def test_extract_requirements():
    """Test requirement extraction from questions."""
    print("Testing extract_answer_requirements...")
    
    # Test thousand hours detection
    req = extract_answer_requirements(
        "How many thousand hours would it take? Round to nearest 1000."
    )
    assert req['unit_scale'] == 'thousands', f"Expected 'thousands', got {req['unit_scale']}"
    assert req['precision'] == 1000, f"Expected 1000, got {req['precision']}"
    print(f"  ✓ Thousand hours: {req}")
    
    # Test numeric only
    req = extract_answer_requirements("Just give the number.")
    assert req['numeric_only'] == True
    print(f"  ✓ Numeric only: {req}")
    
    # Test alphabetical list
    req = extract_answer_requirements("List them alphabetically, comma separated.")
    assert req['order'] == 'alphabetical'
    assert req['format_type'] == 'list'
    print(f"  ✓ Alphabetical list: {req}")
    
    # Test first name only
    req = extract_answer_requirements("Give only the first name.")
    assert req['specificity'] == 'first_name'
    print(f"  ✓ First name only: {req}")
    
    print("  All requirement extraction tests passed!\n")

def test_validate_format():
    """Test answer format validation."""
    print("Testing validate_answer_format...")
    
    # Test thousand scaling correction
    req = {'unit_scale': 'thousands', 'format_type': 'text', 'separator': ', ', 
           'order': None, 'precision': None, 'specificity': None, 'numeric_only': False}
    is_valid, corrected, issues = validate_answer_format("17000", "thousand hours", req)
    assert corrected == "17", f"Expected '17', got '{corrected}'"
    print(f"  ✓ Thousand scaling: 17000 -> {corrected}, issues: {issues}")
    
    # Test alphabetical sorting
    req = {'unit_scale': None, 'format_type': 'list', 'separator': ', ',
           'order': 'alphabetical', 'precision': None, 'specificity': None, 'numeric_only': False}
    is_valid, corrected, issues = validate_answer_format("zebra, apple, mango", "alphabetical list", req)
    assert corrected == "apple, mango, zebra", f"Expected sorted, got '{corrected}'"
    print(f"  ✓ Alphabetical sort: {corrected}")
    
    # Test numeric extraction
    req = {'unit_scale': None, 'format_type': 'numeric', 'separator': ', ',
           'order': None, 'precision': None, 'specificity': None, 'numeric_only': True}
    is_valid, corrected, issues = validate_answer_format("The answer is 42 meters", "just the number", req)
    assert corrected == "42", f"Expected '42', got '{corrected}'"
    print(f"  ✓ Numeric extraction: {corrected}")
    
    print("  All validation tests passed!\n")

def test_normalize():
    """Test answer normalization."""
    print("Testing normalize_answer...")
    
    # Test compound word normalization
    result = normalize_answer("The sea gull flew away.")
    assert "seagull" in result.lower(), f"Expected 'seagull', got '{result}'"
    print(f"  ✓ Compound word: 'sea gull' -> '{result}'")
    
    # Test trailing punctuation
    result = normalize_answer("Paris.")
    assert result == "Paris", f"Expected 'Paris', got '{result}'"
    print(f"  ✓ Trailing punctuation: 'Paris.' -> '{result}'")
    
    # Test whitespace normalization
    result = normalize_answer("  multiple   spaces  ")
    assert result == "multiple spaces", f"Expected 'multiple spaces', got '{result}'"
    print(f"  ✓ Whitespace: '{result}'")
    
    print("  All normalization tests passed!\n")

def test_preprocess():
    """Test question preprocessing."""
    print("Testing preprocess_question...")
    
    # Test reversed text detection
    reversed_q = ".rewsna eht sa tfel fo etisoppo eht etirw ,siht dnatsrednu uoy fI"
    result = preprocess_question(reversed_q)
    # Should detect and reverse
    assert "reversed" in result.lower() or "If you understand" in result, f"Did not detect reversed: {result[:50]}"
    print(f"  ✓ Reversed text detection")
    
    # Normal question should pass through unchanged
    normal_q = "What is the capital of France?"
    result = preprocess_question(normal_q)
    assert result == normal_q, f"Normal question was modified: {result}"
    print(f"  ✓ Normal question unchanged")
    
    print("  All preprocessing tests passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing GAIA Agent Enhancements")
    print("=" * 60 + "\n")
    
    test_extract_requirements()
    test_validate_format()
    test_normalize()
    test_preprocess()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
