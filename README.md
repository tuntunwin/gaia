# GAIA Agent

A general AI assistant agent built with Hugging Face smolagents to answer questions from the GAIA (General AI Assistants) benchmark, specifically Level 1 questions.

## Overview

This project implements an AI agent using the [smolagents](https://huggingface.co/learn/agents-course/unit2/smolagents/why_use_smolagents) framework from Hugging Face. The agent is designed to handle questions from the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA), which evaluates AI assistants on their ability to perform reasoning, web search, and information retrieval tasks.

## Features

- 🤖 AI agent powered by Hugging Face smolagents
- 🔍 Built-in web search capabilities using DuckDuckGo
- 🌐 Web page visiting and content extraction
- 📊 GAIA benchmark dataset integration
- 🎯 Optimized for Level 1 questions (basic reasoning and retrieval)
- 📈 Evaluation tools with accuracy metrics
- 🖥️ Command-line interface for easy usage

## Project Structure

```
gaia/
├── src/
│   └── gaia_agent/
│       ├── __init__.py
│       ├── agent.py         # Core agent implementation
│       ├── dataset.py       # GAIA dataset loader and evaluator
│       └── cli.py           # Command-line interface
├── tests/                   # Test files
├── examples.py             # Example usage scripts
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager (recommended)
- Hugging Face account and API token ([get one here](https://huggingface.co/settings/tokens))

### Setup

1. Clone the repository:

```bash
git clone https://github.com/tuntunwin/gaia.git
cd gaia
```

2. Install uv (if not already installed):

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
uv pip install -e .
```

4. Set your Hugging Face API token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or create a `.env` file:

```bash
	echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## Usage

### Command-Line Interface

#### Ask a single question:

```bash
uv run gaia-agent --question "What is the capital of France?"
```

#### Evaluate on GAIA benchmark (Level 1 validation set):

```bash
uv run gaia-agent --evaluate --split validation --level 1
```

#### Evaluate with limited questions (for testing):

```bash
uv run gaia-agent --evaluate --level 1 --max-questions 5
```

#### Use a specific model:

```bash
uv run gaia-agent --question "Who wrote Romeo and Juliet?" --model "meta-llama/Llama-3.1-70B-Instruct"
```

### Python API

#### Basic usage:

```python
from gaia_agent.agent import GAIAAgent

# Initialize the agent
agent = GAIAAgent()

# Ask a question
answer = agent.run("What is the capital of France?")
print(answer)
```

#### Working with GAIA dataset:

```python
from gaia_agent.agent import GAIAAgent
from gaia_agent.dataset import GAIADatasetLoader

# Load Level 1 questions
loader = GAIADatasetLoader()
questions = loader.get_level_1_questions(split="validation")

# Initialize agent
agent = GAIAAgent()

# Answer a question
result = agent.answer_gaia_question(questions[0])
print(f"Answer: {result['answer']}")
```

#### Evaluation:

```python
from gaia_agent.agent import GAIAAgent
from gaia_agent.dataset import GAIADatasetLoader, GAIAEvaluator

# Load questions and initialize
loader = GAIADatasetLoader()
questions = loader.get_level_1_questions(split="validation")
agent = GAIAAgent()
evaluator = GAIAEvaluator()

# Evaluate
results = []
for q in questions[:5]:  # Test on first 5 questions
    result = agent.answer_gaia_question(q)
  
    # Check if correct
    if "Final answer" in q:
        correct = evaluator.evaluate_answer(result["answer"], q["Final answer"])
        result["correct"] = correct
  
    results.append(result)

# Calculate accuracy
accuracy = evaluator.calculate_accuracy(results)
print(f"Accuracy: {accuracy:.2%}")
```

### Example Scripts

Run the example script to see the agent in action:

```bash
uv run python examples.py
```

## GAIA Benchmark

The GAIA (General AI Assistants) benchmark consists of questions at three difficulty levels:

- **Level 1**: Basic questions requiring simple reasoning and web search (this project's focus)
- **Level 2**: Moderate difficulty with more complex reasoning
- **Level 3**: Advanced questions requiring multi-step reasoning and tool use

Each question may include:

- A text question
- Optional file attachments (PDFs, images, spreadsheets, audio)
- A ground truth answer for validation

## How It Works

The GAIA Agent uses smolagents' `CodeAgent`, which:

1. **Receives a question** from the user or dataset
2. **Plans actions** using the language model
3. **Executes tools** like web search or webpage visiting
4. **Reasons about results** to formulate an answer
5. **Returns the final answer**

The agent has access to:

- `DuckDuckGoSearchTool`: Search the web
- `VisitWebpageTool`: Visit and extract content from webpages
- Python code execution for calculations and data processing

## Configuration

You can customize the agent by:

### Using a different model:

```python
agent = GAIAAgent(model_id="meta-llama/Llama-3.1-70B-Instruct")
```

### Adding custom tools:

```python
from smolagents import tool

@tool
def custom_calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

agent = GAIAAgent(additional_tools=[custom_calculator])
```

## Development

### Running tests:

```bash
uv run pytest tests/
```

### Code style:

```bash
# Format code
uv run black src/

# Type checking
uv run mypy src/
```

## Limitations

- Level 1 questions only (Level 2 and 3 support can be added)
- File attachments are noted but not processed (requires additional tools)
- Web search may be rate-limited
- Answers are evaluated with simple string matching (advanced evaluation needed for complex answers)

## Future Enhancements

- [ ] Support for Level 2 and 3 questions
- [ ] File processing tools (PDF, images, spreadsheets, audio)
- [ ] More sophisticated answer evaluation
- [ ] Caching and rate limiting for API calls
- [ ] Web UI for interactive usage
- [ ] Integration with more LLM providers

## Resources

- [Hugging Face smolagents Documentation](https://huggingface.co/learn/agents-course/unit2/smolagents/why_use_smolagents)
- [GAIA Benchmark Dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [GAIA Paper](https://arxiv.org/abs/2311.12983)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project or the GAIA benchmark, please cite:

```bibtex
@article{gaia2023,
  title={GAIA: A Benchmark for General AI Assistants},
  author={Mialon, Gr\'egoire and others},
  journal={arXiv preprint arXiv:2311.12983},
  year={2023}
}
```
