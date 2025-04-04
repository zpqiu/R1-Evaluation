import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from util.task_handlers import *
from util.model_utils import *
from openai import OpenAI
import concurrent.futures
from functools import partial

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def fetch_response_openai(llm, model_name, max_tokens, temp, messages):
    response = llm.chat.completions.create(
        model=model_name,
        messages=messages,
        n=1,
        temperature=temp,
        max_tokens=max_tokens,
        timeout=10000,
    )
    return response


R1_SYS_V2 = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. "
    "{prompt}\n"
    "Assistant: <think>"
)

def fetch_r1_response(llm, model_name, max_tokens, temp, messages):
    prompt = R1_SYS_V2.format(prompt=messages[1]["content"])

    response = llm.completions.create(
        model=model_name,
        prompt=prompt,
        n=1,
        temperature=temp,
        max_tokens=max_tokens,
        timeout=10000,
    )
    return response


def perform_inference_and_check(handler: TaskHandler, temperature, max_tokens, result_file, llm, system_prompt, args):
    results = handler.load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=False, args=args)
    remaining_data = handler.process_remaining_data(train_data, results)
    conversations = handler.make_conversations(remaining_data, system_prompt, args.model)
    repeated_conversations = []
    repeated_remaining_data = []
    for problem, conversation in zip(remaining_data, conversations):
        for _ in range(args.n):
            repeated_conversations.append(conversation.copy())
            repeated_remaining_data.append(problem.copy())
        
    fetch_partial = partial(fetch_r1_response, llm, args.model, max_tokens, temperature)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
        responses = list(e.map(fetch_partial, repeated_conversations))
        
    total_correct = 0 
    total_finish = 0
    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {}
        token_usages = {}
        for idx, response in enumerate(responses):
            response_str = response.choices[0].text.strip()
            future_to_task[executor.submit(handler.update_results, repeated_remaining_data[idx], response_str, None)] = idx
            
            if idx not in token_usages:
                token_usages[idx] = []
            token_usages[idx].append({
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens
            })

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Generations"):
            idx = future_to_task[future]
            response_entry = future.result()
            total_correct += response_entry["correctness"]
            total_finish += 1

            problem_key = repeated_remaining_data[idx][handler.get_question_key()]
            if problem_key not in results:
                results[problem_key] = repeated_remaining_data[idx]
                if isinstance(handler, NUMINATaskHandler):
                    results[problem_key]["messages"] = ""
                results[problem_key]["responses"] = []
                results[problem_key]["token_usages"] = []
                # prompt = repeated_conversations[idx][1]["content"]
                # results[problem_key]["prompt"] = prompt

            results[problem_key]["responses"].append(response_entry)
            results[problem_key]["token_usages"].extend(token_usages[idx])
                
        print(f"Final acc: {total_correct}/{total_finish}")
        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        print(json.dumps({"acc": acc}))

    completion_tokens = [
        token_usages.get("completion_tokens", 0)
        for key in results for token_usages in results[key].get("token_usages", [])
    ]
    prompt_tokens = [
        token_usages.get("prompt_tokens", 0)
        for key in results for token_usages in results[key].get("token_usages", [])
    ]

    # Token usage summary
    result_dir, result_name = os.path.split(result_file)
    token_usage_dir = os.path.join(result_dir, "token_usage")
    os.makedirs(token_usage_dir, exist_ok=True)

    # Construct the token usage result file path
    token_usage_result_file = os.path.join(token_usage_dir, result_name)

    # Prepare the token usage dictionary
    token_dict = {
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3) if completion_tokens else 0,
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0,
    }

    # Save the token usage dictionary to the result file
    with open(token_usage_result_file, "w") as f:
        json.dump(token_dict, f, indent=4)

    print(f"Token usage saved to {token_usage_result_file}")
    
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="Unified inference and checking for different datasets/tasks.")
    parser.add_argument("--dataset", type=str, required=True, choices=["NUMINA", "APPS", "TACO", "MATH500", "AIME", "GPQADiamond", "MMLU", "MMLUPro", "LiveCodeBench", "GSM8K", "ARC-C"], help="Dataset to process.")
    parser.add_argument("--model", type=str, required=True, default="Qwen/QwQ-32B-Preview", help="The model to run.")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
    parser.add_argument("--split", type=str, default="train", help="Split to use for apps (e.g., train, test).")
    parser.add_argument("--source", type=str, help="Source for the dataset.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=-1, help="End index.")
    parser.add_argument("--result-dir", type=str, default="./", help="Result dir to save files.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling.")
    parser.add_argument("--n", type=int, default=1, help="Number of samples generated per problem.")
    parser.add_argument("--sample-method", type=str, default="random", choices=["random", "uniform_by_difficulty"], help="Method to sample problems.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of problems to sample.")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="Base URL for DeepSeek API.")
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
    result_file = os.path.join(args.result_dir, f"{MODEL_TO_NAME[args.model]}_{args.dataset}_{args.split}_{args.source}.json")

    llm = OpenAI(base_url=args.base_url)
    system_prompt = SYSTEM_PROMPT[args.model]
    perform_inference_and_check(handler, args.temperature, max_tokens, result_file, llm, system_prompt, args)

if __name__ == "__main__":
    main()
