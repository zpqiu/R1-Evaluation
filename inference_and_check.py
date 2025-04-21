import os
import json
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from util.task_handlers import TaskHandler, TASK_HANDLERS
from openai import OpenAI
import concurrent.futures
from functools import partial

# 禁用 HTTP 请求日志
os.environ["OPENAI_LOG"] = "none"
os.environ["URLLIB3_LOG"] = "none"
os.environ["HTTPCORE_LOG"] = "none"
os.environ["HTTPX_LOG"] = "none"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置 OpenAI 和 urllib3 的日志级别为 WARNING
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def fetch_response_openai(llm, model_name, max_tokens, temp, messages):
    try:
        response = llm.chat.completions.create(
            model=model_name,
            messages=messages,
            n=1,
            temperature=temp,
            max_tokens=max_tokens,
            timeout=10000
        )
        return response
    except Exception as e:
        logger.error(f"Error in fetch_response_openai: {str(e)}")
        raise


def fetch_r1_response(llm, prompt_template, model_name, max_tokens, temp, messages):
    try:
        prompt = prompt_template.format(messages[1]["content"])
        response = llm.completions.create(
            model=model_name,
            prompt=prompt,
            n=1,
            temperature=temp,
            max_tokens=max_tokens,
            timeout=10000
        )
        return response
    except Exception as e:
        logger.error(f"Error in fetch_r1_response: {str(e)}")
        raise


def perform_inference_and_check(handler: TaskHandler, temperature, max_tokens, result_file, llm, system_prompt, args):
    try:
        logger.info(f"Starting inference with temperature={temperature}, max_tokens={max_tokens}")
        results = handler.load_existing_results(result_file)
        logger.info(f"Loaded {len(results)} existing results.")
        
        # 首先计算已有结果的准确率
        total_correct = 0
        total_finish = 0
        for problem_key, problem_data in results.items():
            if "responses" in problem_data:
                for response in problem_data["responses"]:
                    if "correctness" in response:
                        total_correct += response["correctness"]
                        total_finish += 1
        
        logger.info(f"Existing results: {total_correct} correct out of {total_finish}")
        
        train_data = handler.load_and_filter_dataset(split=args.split, args=args)
        remaining_data = handler.process_remaining_data(train_data, results)
        conversations = handler.make_conversations(remaining_data, system_prompt, args.model)
        
        logger.info(f"Processing {len(remaining_data)} problems")
        
        repeated_conversations = []
        repeated_remaining_data = []
        for problem, conversation in zip(remaining_data, conversations):
            for _ in range(args.n):
                repeated_conversations.append(conversation.copy())
                repeated_remaining_data.append(problem.copy())
        
        fetch_partial = partial(fetch_r1_response, llm, args.prompt, args.model, max_tokens, temperature)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
            responses = list(e.map(fetch_partial, repeated_conversations))
        
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_task = {}
            token_usages = {}
            for idx, response in enumerate(responses):
                try:
                    response_str = response.choices[0].text.strip()
                    future_to_task[executor.submit(handler.update_results, repeated_remaining_data[idx], response_str, None)] = idx
                    
                    if idx not in token_usages:
                        token_usages[idx] = []
                    token_usages[idx].append({
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens
                    })
                except Exception as e:
                    logger.error(f"Error processing response {idx}: {str(e)}")
                    continue

            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Generations"):
                try:
                    idx = future_to_task[future]
                    response_entry = future.result()
                    total_correct += response_entry["correctness"]
                    total_finish += 1

                    problem_key = repeated_remaining_data[idx][handler.get_question_key()]
                    if problem_key not in results:
                        results[problem_key] = repeated_remaining_data[idx]
                        results[problem_key]["responses"] = []
                        results[problem_key]["token_usages"] = []

                    results[problem_key]["responses"].append(response_entry)
                    results[problem_key]["token_usages"].extend(token_usages[idx])
                except Exception as e:
                    logger.error(f"Error processing future result: {str(e)}")
                    continue
                    
            logger.info(f"Final accuracy: {total_correct}/{total_finish}")
            acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
            # 使用特殊格式输出结果，便于 eval.py 解析
            print(f"EVAL_RESULT: {json.dumps({'acc': acc})}")

        # 保存结果
        try:
            save_results(results, result_file)
            save_token_usage(results, result_file)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in perform_inference_and_check: {str(e)}")
        raise

def save_results(results, result_file):
    """保存结果到文件"""
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    logger.info(f"Results saved to {result_file}")

def save_token_usage(results, result_file):
    """保存token使用统计"""
    completion_tokens = [
        token_usages.get("completion_tokens", 0)
        for key in results for token_usages in results[key].get("token_usages", [])
    ]
    prompt_tokens = [
        token_usages.get("prompt_tokens", 0)
        for key in results for token_usages in results[key].get("token_usages", [])
    ]

    result_dir, result_name = os.path.split(result_file)
    token_usage_dir = os.path.join(result_dir, "token_usage")
    os.makedirs(token_usage_dir, exist_ok=True)

    token_usage_result_file = os.path.join(token_usage_dir, result_name)

    token_dict = {
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3) if completion_tokens else 0,
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0,
    }

    with open(token_usage_result_file, "w") as f:
        json.dump(token_dict, f, indent=4)

    logger.info(f"Token usage saved to {token_usage_result_file}")


def main():
    parser = argparse.ArgumentParser(description="Unified inference and checking for different datasets/tasks.")
    parser.add_argument("--dataset", type=str, required=True, choices=["NUMINA", "APPS", "TACO", "MATH500", "AIME", "GPQADiamond", "MMLU", "MMLUPro", "LiveCodeBench", "GSM8K", "ARC-C"], help="Dataset to process.")
    parser.add_argument("--model", type=str, required=True, default="Qwen/QwQ-32B-Preview", help="The model to run.")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
    parser.add_argument("--split", type=str, default="train", help="Split to use for apps (e.g., train, test).")
    parser.add_argument("--result-dir", type=str, default="./", help="Result dir to save files.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling.")
    parser.add_argument("--n", type=int, default=1, help="Number of samples generated per problem.")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="Base URL for DeepSeek API.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for the model.")
    args = parser.parse_args()
    
    handler: TaskHandler = TASK_HANDLERS[args.dataset]()

    print(f"Temperature: {args.temperature}")
    max_tokens = args.max_tokens
    if args.temperature == 0 and args.n > 1:
        args.n = 1
        print("Warning: Temperature 0 does not support multiple samples. Setting n=1.")

    # create result dir if not exists
    if args.result_dir and not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    result_file = os.path.join(args.result_dir, f"{args.model}_{args.dataset}_t{args.temperature}_n{args.n}.json")

    llm = OpenAI(base_url=args.base_url)
    perform_inference_and_check(handler, args.temperature, max_tokens, result_file, llm, "", args)

if __name__ == "__main__":
    main()
