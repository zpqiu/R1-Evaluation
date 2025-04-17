import os
import argparse
import subprocess
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

if os.environ.get("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = "sk-1234567890"

@dataclass
class EvalResult:
    greedy_acc: float
    avg_acc: float | None = None
    n_samples: int | None = None

class Evaluator:
    def __init__(self, config_path: str, output_dir: str):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = self._load_prompt_file(self.config["prompt_file"])
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def _load_prompt_file(
        self, prompt_file: Optional[os.PathLike],
    ) -> Optional[str]:
        """Load prompt from file if it exists, otherwise return as is."""
        if prompt_file is None:
            return None
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Prompt file {prompt_file} not found")

    def _run_eval(self, dataset: str, params: Dict[str, Any], is_greedy: bool) -> float:
        script_path = "inference_and_check.py"
        command = [
            "python", script_path,
            "--model", self.config["model"],
            "--dataset", dataset,
            "--split", params["split"],
            "--base-url", self.config["base_url"],
            "--max_tokens", str(params["max_response_length"]),
            "--temperature", "0" if is_greedy else str(self.config["temperature"]),
            "--n", "1" if is_greedy else str(params["n"]),
            "--result-dir", self.output_dir,
            "--sample-size", "-1",
            "--prompt", self.prompt,
        ]

        print(f"Running eval for {dataset} ({'greedy' if is_greedy else 'sampling'})")
        try:
            output = subprocess.check_output(command, text=True)
            accuracy = self._extract_accuracy(output)
            return accuracy if accuracy is not None else 0.0
        except subprocess.CalledProcessError as e:
            print(f"Error running eval for {dataset}: {e}")
            return 0.0

    def _extract_accuracy(self, output: str) -> float | None:
        for line in reversed(output.splitlines()):
            try:
                data = json.loads(line.replace("'", '"'))
                if "acc" in data:
                    return data["acc"]
            except json.JSONDecodeError:
                continue
        return None

    def run_all_evals(self) -> Dict[str, EvalResult]:
        results = {}
        for dataset, params in self.config["datasets"].items():
            # 运行 greedy (pass@1)
            greedy_acc = self._run_eval(dataset, params, is_greedy=True)
            
            # 运行 sampling (avg@n)
            avg_acc = self._run_eval(dataset, params, is_greedy=False)
            
            results[dataset] = EvalResult(
                greedy_acc=greedy_acc,
                avg_acc=avg_acc,
                n_samples=params["n"]
            )
            
        return results

    def save_results(self, results: Dict[str, EvalResult]):
        # 保存详细结果到JSON
        detailed_results = {
            dataset: {
                "pass@1": result.greedy_acc,
                f"avg@{result.n_samples}": result.avg_acc,
                "n_samples": result.n_samples
            } for dataset, result in results.items()
        }
        
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # 生成漂亮的结果表格
        self._print_results_table(results)

    def _print_results_table(self, results: Dict[str, EvalResult]):
        print("\n" + "="*60)
        print(f"{'Dataset':<20} {'Pass@1':>10} {'Avg@N':>10} {'N':>5}")
        print("-"*60)
        
        for dataset, result in results.items():
            print(f"{dataset:<20} {result.greedy_acc:>10.2%} {result.avg_acc:>10.2%} {result.n_samples:>5}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Run evaluations based on config file")
    parser.add_argument("--config", default="config.json", type=str, help="Path to config file")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save results")
    args = parser.parse_args()

    evaluator = Evaluator(args.config, args.output_dir)
    results = evaluator.run_all_evals()
    evaluator.save_results(results)

if __name__ == "__main__":
    main()
