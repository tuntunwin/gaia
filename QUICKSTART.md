# Quick Start Guide

This guide will help you get started with the GAIA Agent quickly.

## Prerequisites

- Python 3.8 or higher
- Hugging Face account (free)

## Step 1: Get Your Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read permissions are sufficient)
3. Copy the token

## Step 2: Install the Package

```bash
# Clone the repository
git clone https://github.com/tuntunwin/gaia.git
cd gaia

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Step 3: Set Your API Token

```bash
# Option 1: Environment variable (temporary, current session only)
export HF_TOKEN="your_token_here"

# Option 2: Create a .env file (persistent)
echo "HF_TOKEN=your_token_here" > .env
```

## Step 4: Verify Installation

```bash
# Run verification script
python verify.py

# Check CLI is working
gaia-agent --help
```

## Step 5: Try Your First Query

```bash
# Ask a simple question
gaia-agent --question "What is the capital of France?"
```

## Step 6: Evaluate on GAIA Benchmark

```bash
# Run on a few Level 1 questions (for testing)
gaia-agent --evaluate --level 1 --max-questions 3

# Run on all Level 1 validation questions
gaia-agent --evaluate --level 1 --split validation
```

## Python API Usage

```python
from gaia_agent.agent import GAIAAgent

# Initialize agent
agent = GAIAAgent()

# Ask a question
answer = agent.run("What year was the Python programming language created?")
print(answer)
```

## Working with GAIA Dataset

```python
from gaia_agent.agent import GAIAAgent
from gaia_agent.dataset import GAIADatasetLoader, GAIAEvaluator

# Load dataset
loader = GAIADatasetLoader()
questions = loader.get_level_1_questions(split="validation")

# Initialize agent
agent = GAIAAgent()
evaluator = GAIAEvaluator()

# Answer first question
question_data = questions[0]
result = agent.answer_gaia_question(question_data)

# Evaluate
if "Final answer" in question_data:
    correct = evaluator.evaluate_answer(
        result["answer"], 
        question_data["Final answer"]
    )
    print(f"Correct: {correct}")
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'ddgs'"

Install the missing dependency:
```bash
pip install ddgs
```

### "No Hugging Face API token provided"

Make sure you've set the `HF_TOKEN` environment variable or created a `.env` file with your token.

### Rate Limiting

If you encounter rate limiting with DuckDuckGo search:
- Reduce the number of questions being evaluated
- Add delays between queries
- Consider using different search tools

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [examples.py](examples.py) for more usage examples
- Review the [tests/](tests/) directory for testing examples
- Modify the agent with custom tools for your specific needs

## Need Help?

- Check existing issues on GitHub
- Create a new issue with details about your problem
- Review the [Hugging Face smolagents documentation](https://huggingface.co/learn/agents-course/unit2/smolagents/why_use_smolagents)
