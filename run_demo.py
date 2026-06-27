from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.dataset import load_bfcl_jsonl, split_examples
from src.harness import AgentHarness
from src.model_adapter import OpenAICompatibleAdapter, RuleBasedAdapter
from src.skill_store import SkillStore
from src.reflector import LLMFailureReflector, RuleFailureReflector


ROOT = Path(__file__).resolve().parent
PROGRESS_WIDTH = 100


def build_model(name: str):
    if name == "rule":
        return RuleBasedAdapter()
    if name == "openai":
        return OpenAICompatibleAdapter()
    raise ValueError(f"Unknown model adapter: {name}")


def validate_runtime_config(model_name: str, reflector_name: str) -> None:
    """fail-fast 常见的API配置错误检查。"""

    needs_key = model_name == "openai" or reflector_name == "llm" or (reflector_name == "auto" and model_name == "openai")
    if not needs_key:
        return

    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
    print(f"OpenAI-compatible endpoint: {base_url}", flush=True)
    print(f"OpenAI-compatible model: {model}", flush=True)
    print(f"API key configured: {has_key}", flush=True)
    if not has_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY or DEEPSEEK_API_KEY before running --model openai."
        )


def print_progress_line(text: str, end: str = "\n") -> None:
    """打印进度行，确保不会换行。"""

    print(text.ljust(PROGRESS_WIDTH), end=end, flush=True)


def cli_progress(event: dict) -> None:
    """callback 打印进度信息到命令行。事件字典包含以下键：
    - event: 事件类型，如 iteration_start, batch_start, batch_item, batch_end, stage, iteration_end
    - iteration: 当前迭代次数
    - stage: 当前阶段名称
    - current: 当前处理的案例数
    - total: 总案例数
    - concurrency: 并发数
    """

    event_type = event.get("event")
    iteration = event.get("iteration")
    stage = event.get("stage", "")
    current = event.get("current", 0)
    total = event.get("total", 0)

    prefix = f"[Iteration {iteration}] " if iteration is not None else ""
    if event_type == "iteration_start":
        print_progress_line(f"\n{prefix}Starting closed-loop iteration")
    elif event_type == "batch_start":
        print_progress_line(f"{prefix}{stage}: 0/{total} (concurrency={event.get('concurrency', '-')})", end="")
    elif event_type == "batch_item":
        print_progress_line(f"\r{prefix}{stage}: {current}/{total} (concurrency={event.get('concurrency', '-')})", end="")
    elif event_type == "batch_end":
        print_progress_line(f"\r{prefix}{stage}: {total}/{total} done")
    elif event_type == "stage":
        print_progress_line(f"{prefix}{stage}...")
    elif event_type == "iteration_end":
        print_progress_line(f"{prefix}Iteration finished accepted={event.get('accepted')}")


def build_reflector(name: str, model_name: str):
    """根据名称和模型类型构建适当的 reflector。"""

    if name == "rule":
        return RuleFailureReflector()
    if name == "llm":
        return LLMFailureReflector()
    if name == "auto":
        return LLMFailureReflector() if model_name == "openai" else RuleFailureReflector()
    raise ValueError(f"Unknown reflector: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 BFCL 风格的 Agent Harness 自进化演示。")
    parser.add_argument("--data", default=str(ROOT / "data" / "mini_bfcl_subset.jsonl"), help="BFCL 兼容的 JSONL 数据路径。")
    parser.add_argument("--iterations", type=int, default=3, help="进化迭代次数。")
    parser.add_argument("--model", choices=["rule", "openai"], default="rule", help="模型适配器。")
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "latest"), help="报告输出目录。")
    parser.add_argument("--reset-skill", action="store_true", help="运行前重置 Skill.md 和 fewshots。")
    parser.add_argument("--concurrency", type=int, default=10, help="并发模型调用数。默认：10。")
    parser.add_argument("--reflector", choices=["auto", "rule", "llm"], default="auto", help="反思策略。auto 会在 --model openai 时使用 LLM，在 --model rule 时使用规则。")
    args = parser.parse_args()

    validate_runtime_config(args.model, args.reflector)
    examples = load_bfcl_jsonl(args.data)
    optimize_examples, regression_examples = split_examples(examples)

    skill_store = SkillStore(ROOT / "skills" / "tool_calling")
    if args.reset_skill:
        skill_store.reset()

    harness = AgentHarness(
        model=build_model(args.model),
        skill_store=skill_store,
        run_dir=args.run_dir,
        concurrency=args.concurrency,
        reflector=build_reflector(args.reflector, args.model),
        progress_callback=cli_progress,
    )
    reports = harness.run(optimize_examples, regression_examples, iterations=args.iterations)

    print("\nBFCL Agent Harness Demo finished.")
    for report in reports:
        before = report.metrics_before["exact_match_rate"]
        after = report.metrics_after["exact_match_rate"] if report.metrics_after else None
        details = report.gate_details
        print(
            f"Iteration {report.iteration}: accepted={report.accepted} "
            f"opt_exact={before} regression_exact={after} reason={report.gate_reason}"
        )
        if details:
            print(f"  new_rules={len(details.get('new_rules', []))} bad_cases={details.get('bad_case_count')}")
            print(
                f"  success_cases={details.get('success_case_count')} "
                f"scoped_rules={len(details.get('scoped_rules', []))} "
                f"candidate_fewshots={details.get('candidate_fewshot_count')}"
            )
            print(f"  diagnosis_distribution={details.get('diagnosis_distribution')}")
            if "improvement" in details:
                regression_diff = details.get("regression_diff", {})
                print(
                    "  gate_metrics="
                    f"before={details['before_exact_match_rate']:.4f}, "
                    f"candidate={details['after_exact_match_rate']:.4f}, "
                    f"deployed={details.get('deployed_exact_match_rate', details['before_exact_match_rate']):.4f}, "
                    f"delta={details['improvement']:.4f}, "
                    f"drop={details['regression_drop']:.4f}"
                )
                print(
                    "  regression_diff="
                    f"improved={regression_diff.get('improved_count', 0)}, "
                    f"regressed={regression_diff.get('regressed_count', 0)}, "
                    f"changed={regression_diff.get('changed_count', 0)}"
                )
            if details.get("experience_path"):
                print(f"  experience={details.get('experience_path')}")
    print(f"\nReports saved to: {Path(args.run_dir).resolve()}")


if __name__ == "__main__":
    main()
