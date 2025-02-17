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
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 30000 --tp 2
# and set the OPENAI_API_KEY
export OPENAI_API_KEY=xys
```
- Login Hugging Face
```bash
huggingface-cli login
```

### Step 2: Add New Model

Add new model to `utils/model_utils.py` by changing `MODEL_TO_NAME` and `MODEL_TO_SYS`


### Step 3: Run Inference and Check

```bash
uv run eval.py --model Qwen/Qwen2.5-3B-Instruct --evals=AIME,MATH500,GPQADiamond --base-url http://localhost:30000/v1 --output_file=Qwen2.5-3B-Instruct.txt  --temperature 0.6 --n 8 --sample-size 100
```

`n` is the number of samples for each problem. The result metric is Pass@1.


## Acknowledgement

This project is inspired by [SkyThought](https://github.com/NovaSky-AI/SkyThought).
