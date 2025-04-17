# Evaluation Tool for Reasoning Models

## Installation

This project use uv to manage dependencies. Please refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

After installing uv, you can install the dependencies by running:
```bash
uv sync
```

## Usage

### Step 1: Environment Setup

- Load OpenAI Compatible API HTTP Endpoint
```bash
python3 -m sglang.launch_server --model-path pe-nlp/Qwen2.5-7b-grpo-orz-cl2-step150 --host 0.0.0.0 --port 30000 --tp 2
```
- Login Hugging Face
```bash
huggingface-cli login
```

### Step 2: Quick Start (Default Evaluation)

Run evaluation with default settings:
```bash
uv run eval.py --output-dir results
```

This will:
1. Use default configuration (see `config.json` for details)
2. Run greedy decoding (Pass@1) for each dataset
3. Run sampling (Avg@N) for each dataset
4. Save detailed results to `results/detailed_results.json`
5. Print a summary table of all results

### Step 3: Custom Configuration

If you want to customize the evaluation, create a config file (e.g., `config.json`) with the following structure:
```json
{
    "model": "path/to/your/model",
    "base_url": "http://localhost:30000/v1",
    "temperature": 0.6,
    "prompt_file": "path/to/prompt.txt",
    "datasets": {
        "dataset_name": {
            "split": "test",
            "max_response_length": 2048,
            "n": 8
        }
    }
}
```

Then run with your custom config:
```bash
uv run eval.py --config config.json --output-dir results
```

## Acknowledgement

This project is inspired by [SkyThought](https://github.com/NovaSky-AI/SkyThought).
