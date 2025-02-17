# Evaluation Tool for Reasoning Models

## Usage

1. Add new model to `utils/model_utils.py` by changing `MODEL_TO_NAME` and `MODEL_TO_SYS`

### Step 1: Environment Setup

- Load OpenAI Compatible API HTTP Endpoint
```bash
export OPENAI_API_KEY=xys
```
- Login Hugging Face
```bash
huggingface-cli login
```

### Step 2: Run Inference and Check

```bash
python3 eval.py --model ./R1-3B-3096 --evals=AIME,MATH500,GPQADiamond --base-url http://localhost:30000/v1 --output_file=R1-3B-3096-Reward2.txt  --temperatures 0.0 --n 1 --sample-size -1
```

