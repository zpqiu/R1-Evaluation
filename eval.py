import os
import argparse
import subprocess
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import wandb

# 全局配置
WANDB_PROJECT = "alex_rl_wandb"

if os.environ.get("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = "sk-1234567890"

@dataclass
class EvalResult:
    temperature_results: Dict[float, float]
    n_samples: int | None = None

class Evaluator:
    def __init__(self, config_path: str, output_dir: str):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = self._load_prompt_file(self.config["prompt_file"])
        
        # 初始化wandb，使用output_dir作为run name
        run_name = Path(output_dir).name
        self.run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=self.config
        )
        
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

    def _run_eval(self, dataset: str, params: Dict[str, Any], temperature: float) -> float:
        script_path = "inference_and_check.py"
        # 当temperature=0时，强制设置n=1
        n_samples = 1 if temperature == 0.0 else params["n"]
        
        command = [
            "python", script_path,
            "--model", self.config["model"],
            "--dataset", dataset,
            "--split", params["split"],
            "--base-url", self.config["base_url"],
            "--max_tokens", str(params["max_response_length"]),
            "--temperature", str(temperature),
            "--n", str(n_samples),
            "--result-dir", self.output_dir,
            "--prompt", self.prompt,
        ]

        print(f"Running eval for {dataset} (temperature={temperature}, n={n_samples})")
        try:
            output = subprocess.check_output(command, text=True)
            accuracy = self._extract_accuracy(output)
            return accuracy if accuracy is not None else 0.0
        except subprocess.CalledProcessError as e:
            print(f"Error running eval for {dataset}: {e}")
            return 0.0

    def _extract_accuracy(self, output: str) -> float | None:
        for line in output.splitlines():
            if line.startswith("EVAL_RESULT: "):
                try:
                    data = json.loads(line[12:])  # 去掉 "EVAL_RESULT: " 前缀
                    if "acc" in data:
                        return data["acc"]
                except json.JSONDecodeError:
                    continue
        return None

    def run_all_evals(self) -> Dict[str, EvalResult]:
        results = {}
        for dataset, params in self.config["datasets"].items():
            print(f"\n{'='*20} 开始评估数据集: {dataset} {'='*20}")
            
            # 运行所有温度值的评估
            temperature_results = {}
            for temp in self.config["temperatures"]:
                acc = self._run_eval(dataset, params, temperature=temp)
                print(f"Temperature={temp}: {acc:.2%}")
                temperature_results[temp] = acc
            
            results[dataset] = EvalResult(
                temperature_results=temperature_results,
                n_samples=params["n"]
            )
            
            print(f"{'='*20} 数据集 {dataset} 评估完成 {'='*20}\n")
            
        return results

    def save_results(self, results: Dict[str, EvalResult]):
        # 保存详细结果到JSON
        detailed_results = {
            dataset: {
                "temperature_results": result.temperature_results,
                "n_samples": result.n_samples
            } for dataset, result in results.items()
        }
        
        results_file = self.output_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # 生成漂亮的结果表格
        self._print_results_table(results)

        # 记录结果到wandb
        for dataset, result in results.items():
            for temp, acc in result.temperature_results.items():
                n = result.n_samples if temp != 0.0 else 1
                wandb.log({
                    f"{dataset}/T={temp}/Avg@{n}": acc
                })

        # 上传相关文件到wandb
        if self.config["prompt_file"]:
            wandb.save(self.config["prompt_file"])
        
        # 上传模型生成的responses文件
        for dataset in self.config["datasets"].keys():
            for temp in self.config["temperatures"]:
                n = self.config['datasets'][dataset]['n'] if temp != 0.0 else 1
                response_file = self.output_dir / f"{dataset}_t{temp}_n{n}.json"
                if response_file.exists():
                    wandb.save(str(response_file))

        # 完成wandb运行
        self.run.finish()

    def _print_results_table(self, results: Dict[str, EvalResult]):
        # 获取所有温度值并排序
        all_temps = sorted(set(temp for result in results.values() 
                             for temp in result.temperature_results.keys()))
        
        # 计算列宽
        dataset_width = max(len(dataset) for dataset in results.keys()) + 2
        temp_width = 10  # 每个温度列的宽度
        
        # 打印表头
        header = f"{'Dataset':<{dataset_width}}"
        for temp in all_temps:
            header += f"| T={temp:<6}"
        print("\n" + "=" * (dataset_width + len(all_temps) * (temp_width + 1)))
        print(header)
        print("-" * (dataset_width + len(all_temps) * (temp_width + 1)))
        
        # 打印每行数据
        for dataset, result in results.items():
            row = f"{dataset:<{dataset_width}}"
            for temp in all_temps:
                acc = result.temperature_results.get(temp, 0.0)
                row += f"| {acc:>8.2%}"
            print(row)
        
        print("=" * (dataset_width + len(all_temps) * (temp_width + 1)))

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
