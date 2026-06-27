"""Agent Harness。

闭环流程的设计思路为：
1. 执行 (Execute): 使用当前的 Skill 和 few-shot 示例对一组优化样例进行预测。
2. 评估 (Evaluate): 对预测结果进行评估，计算准确率、错误类型等指标。
3. 挖掘 (Mine): 从评估结果中挖掘失败案例 (Bad Cases) 和成功案例 (Success Cases)。
4. 反思 (Reflect): 使用反思器 (Reflector) 对失败案例进行分析，提出可重用的规则和诊断信息。
5. 提出候选记忆更新 (Propose): 根据反思结果和挖掘的经验，提出候选的 Skill 更新和 few-shot 示例。
6. 回归门控 (Regression Gate): 在回归样例上评估候选 Skill 和 few-shot 示例，确保不会引入新的错误。
7. 接受或回滚 (Accept/Rollback): 根据回归评估结果和门控策略，决定是否接受候选更新，或者回滚到之前的 Skill 版本。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import json
from pathlib import Path
import time
from typing import Any, Callable

from .badcase import BadCaseMiner
from .evaluator import BfclEvaluator
from .experience import ExperienceMiner
from .evolver import SkillEvolver
from .model_adapter import ModelAdapter
from .prompts import build_tool_calling_prompt
from .reflector import Reflector, RuleFailureReflector
from .schema import BfclExample, IterationReport, PredictionRecord
from .skill_store import SkillStore


# 进度回调类型：接收包含事件信息的字典
ProgressCallback = Callable[[dict[str, Any]], None]


class AgentHarness:
    """Agent Harness 掌管整个闭环流程：

    执行 -> 评估 -> 挖掘正/负样本 -> 反思 -> 提出候选记忆更新 -> 回归门控 -> 接受或回滚。
    """

    def __init__(
        self,
        model: ModelAdapter,
        skill_store: SkillStore,
        run_dir: str | Path,
        min_improvement: float = 0.001,
        max_regression_drop: float = 0.05,
        concurrency: int = 10,
        reflector: Reflector | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """初始化 Harness，绑定模型、Skill 存储、运行目录与门控阈值。"""

        self.model = model
        self.skill_store = skill_store
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.min_improvement = min_improvement
        self.max_regression_drop = max_regression_drop
        self.concurrency = max(1, concurrency)
        self.progress_callback = progress_callback

        # 创建子目录用于保存轨迹和经验产物
        self.traces_dir = self.run_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.experience_dir = self.run_dir / "experience"
        self.experience_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各子模块
        self.evaluator = BfclEvaluator()
        self.badcase_miner = BadCaseMiner()
        self.reflector = reflector or RuleFailureReflector()
        self.experience_miner = ExperienceMiner()
        self.evolver = SkillEvolver()

        # 记录当前最优（champion）性能与版本
        self.champion_exact_match: float | None = None
        self.champion_skill_version: str | None = None

    def _emit_progress(self, **payload: Any) -> None:
        """如果注册了进度回调，则发送进度事件。"""

        if self.progress_callback:
            self.progress_callback(payload)

    @staticmethod
    def _safe_stage_name(stage: str) -> str:
        """将阶段名转换为可用于文件名的安全字符串。"""

        return "".join(ch.lower() if ch.isalnum() else "_" for ch in stage).strip("_")

    @staticmethod
    def _calls_to_json(calls: list) -> list[dict]:
        """将函数调用列表序列化为 JSON 友好的字典列表。"""

        return [{"name": call.name, "arguments": call.arguments} for call in calls]

    @staticmethod
    def _tools_to_json(example: BfclExample) -> list[dict]:
        """将样例中的工具定义序列化为 JSON 友好的字典列表。"""

        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in example.tools
        ]

    @staticmethod
    def _skill_diff(before: str, after: str) -> str:
        """为仪表盘生成可读的 Skill 文本差异。"""

        if before == after:
            return ""
        return "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile="SKILL_before.md",
                tofile="SKILL_candidate.md",
                lineterm="",
            )
        )

    def _run_one_example(
        self,
        example: BfclExample,
        skill_text: str,
        fewshots: list[dict],
    ) -> PredictionRecord:
        """对单个样例执行一次模型预测并记录耗时与字符数。"""

        prompt = build_tool_calling_prompt(example, skill_text, fewshots)
        start = time.perf_counter()
        calls, raw_output, parse_error = self.model.generate_calls(example, prompt, skill_text, fewshots)
        latency = time.perf_counter() - start
        return PredictionRecord(
            example=example,
            predicted_calls=calls,
            raw_output=raw_output,
            parse_error=parse_error,
            latency_seconds=latency,
            prompt_chars=len(prompt),
            raw_output_chars=len(raw_output or ""),
        )

    def execute(
        self,
        examples: list[BfclExample],
        skill_text: str | None = None,
        fewshots: list[dict] | None = None,
        stage: str = "execute",
        iteration: int | None = None,
    ) -> list[PredictionRecord]:
        """批量执行样例预测，支持并发与进度回调。"""

        skill_text = skill_text if skill_text is not None else self.skill_store.read_skill()
        fewshots = fewshots if fewshots is not None else self.skill_store.read_fewshots(limit=5)
        records: list[PredictionRecord | None] = [None] * len(examples)

        total = len(examples)
        done = 0
        self._emit_progress(
            event="batch_start",
            iteration=iteration,
            stage=stage,
            total=total,
            current=0,
            concurrency=self.concurrency,
        )

        # 并发为 1 或样例过少时直接串行执行，避免线程池开销
        if self.concurrency == 1 or total <= 1:
            for index, example in enumerate(examples):
                records[index] = self._run_one_example(example, skill_text, fewshots)
                done += 1
                self._emit_progress(
                    event="batch_item",
                    iteration=iteration,
                    stage=stage,
                    total=total,
                    current=done,
                    example_id=example.id,
                    concurrency=self.concurrency,
                )
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                future_to_index = {
                    pool.submit(self._run_one_example, example, skill_text, fewshots): index
                    for index, example in enumerate(examples)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        records[index] = future.result()
                    except Exception as exc:  # noqa: BLE001 - 将 API 失败记录为可追踪错误。
                        example = examples[index]
                        records[index] = PredictionRecord(
                            example=example,
                            predicted_calls=[],
                            raw_output="",
                            parse_error=f"model_call_error: {exc}",
                        )
                    done += 1
                    record = records[index]
                    assert record is not None
                    self._emit_progress(
                        event="batch_item",
                        iteration=iteration,
                        stage=stage,
                        total=total,
                        current=done,
                        example_id=record.example.id,
                        concurrency=self.concurrency,
                    )

        self._emit_progress(
            event="batch_end",
            iteration=iteration,
            stage=stage,
            total=total,
            current=total,
            concurrency=self.concurrency,
        )
        return [record for record in records if record is not None]

    def _save_stage_trace(
        self,
        iteration: int | None,
        stage: str,
        records: list[PredictionRecord],
        scores: list,
        metrics: dict,
        bad_cases: list[dict],
        diagnoses: list[dict] | None = None,
    ) -> str:
        """保存某阶段的执行轨迹、评分、指标和诊断信息到 JSON 文件。"""

        stage_name = self._safe_stage_name(stage)
        iteration_name = "none" if iteration is None else f"{iteration:02d}"
        path = self.traces_dir / f"iteration_{iteration_name}_{stage_name}.json"
        score_by_id = {score.example_id: score for score in scores}
        payload = {
            "iteration": iteration,
            "stage": stage,
            "metrics": metrics,
            "bad_case_count": len(bad_cases),
            "diagnosis_count": len(diagnoses or []),
            "diagnoses": diagnoses or [],
            "records": [],
        }
        for record in records:
            score = score_by_id[record.example.id]
            payload["records"].append(
                {
                    "id": record.example.id,
                    "tags": record.example.tags,
                    "user_query": record.example.user_query,
                    "tools": self._tools_to_json(record.example),
                    "gold_calls": self._calls_to_json(record.example.gold_calls),
                    "predicted_calls": self._calls_to_json(record.predicted_calls),
                    "raw_output": record.raw_output,
                    "parse_error": record.parse_error,
                    "latency_seconds": round(record.latency_seconds, 4),
                    "prompt_chars": record.prompt_chars,
                    "raw_output_chars": record.raw_output_chars,
                    "score": {
                        "exact_match": score.exact_match,
                        "function_match": score.function_match,
                        "argument_match": score.argument_match,
                        "order_match": score.order_match,
                        "json_valid": score.json_valid,
                        "error_types": score.error_types,
                    },
                }
            )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def evaluate_examples(
        self,
        examples: list[BfclExample],
        skill_text: str | None = None,
        fewshots: list[dict] | None = None,
        stage: str = "evaluate",
        iteration: int | None = None,
        save_trace: bool = True,
        diagnoses: list[dict] | None = None,
    ) -> tuple[dict, list[dict], str | None, list[PredictionRecord], list]:
        """执行并评估样例，返回指标、Bad Cases、轨迹路径、记录和评分。"""

        records = self.execute(examples, skill_text=skill_text, fewshots=fewshots, stage=stage, iteration=iteration)
        metrics, scores = self.evaluator.evaluate(records)
        bad_cases = self.badcase_miner.mine(records, scores)
        trace_path = None
        if save_trace:
            trace_path = self._save_stage_trace(iteration, stage, records, scores, metrics, bad_cases, diagnoses)
        return metrics, bad_cases, trace_path, records, scores

    def _save_experience(
        self,
        iteration: int,
        success_cases: list[dict],
        bad_cases: list[dict],
        scoped_rules: list[dict],
        candidate_fewshots: list[dict],
        diagnoses: list[dict],
    ) -> str:
        """保存本次迭代挖掘出的正/负案例、规则与候选 few-shot。"""

        path = self.experience_dir / f"iteration_{iteration:02d}_experience.json"
        payload = {
            "iteration": iteration,
            "success_case_count": len(success_cases),
            "bad_case_count": len(bad_cases),
            "success_cases": success_cases,
            "bad_cases": bad_cases,
            "diagnoses": diagnoses,
            "scoped_rules": scoped_rules,
            "candidate_fewshots": candidate_fewshots,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _infra_failure_summary(diagnoses: list[dict], total_cases: int) -> dict[str, Any]:
        """统计基础设施/模型 API 类失败的摘要信息。"""

        infra = [item for item in diagnoses if item.get("skill_patch_type") == "infrastructure_error"]
        total = max(1, total_cases)
        return {
            "infra_failure_count": len(infra),
            "infra_failure_rate": round(len(infra) / total, 4),
            "sample_infra_errors": [item.get("root_cause", "") for item in infra[:3]],
        }

    def _compare_regression(
        self,
        before_records: list[PredictionRecord],
        before_scores: list,
        after_records: list[PredictionRecord],
        after_scores: list,
    ) -> dict[str, Any]:
        """对比候选 Skill 在回归集上应用前后的样例级变化。"""

        before_record_by_id = {record.example.id: record for record in before_records}
        after_record_by_id = {record.example.id: record for record in after_records}
        before_score_by_id = {score.example_id: score for score in before_scores}
        after_score_by_id = {score.example_id: score for score in after_scores}

        improved: list[dict] = []
        regressed: list[dict] = []
        changed: list[dict] = []
        for case_id, before_score in before_score_by_id.items():
            after_score = after_score_by_id.get(case_id)
            if after_score is None:
                continue
            before_record = before_record_by_id[case_id]
            after_record = after_record_by_id[case_id]
            item = {
                "id": case_id,
                "user_query": before_record.example.user_query,
                "gold_calls": self._calls_to_json(before_record.example.gold_calls),
                "before_predicted_calls": self._calls_to_json(before_record.predicted_calls),
                "after_predicted_calls": self._calls_to_json(after_record.predicted_calls),
                "before_errors": before_score.error_types,
                "after_errors": after_score.error_types,
            }
            if self._calls_to_json(before_record.predicted_calls) != self._calls_to_json(after_record.predicted_calls):
                changed.append(item)
            if not before_score.exact_match and after_score.exact_match:
                improved.append(item)
            elif before_score.exact_match and not after_score.exact_match:
                regressed.append(item)

        return {
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "changed_count": len(changed),
            "improved_examples": improved[:5],
            "regressed_examples": regressed[:5],
            "changed_examples": changed[:5],
        }

    def _build_candidate_fewshots(self, candidate_fewshots: list[dict], base_fewshots: list[dict]) -> list[dict]:
        """将基础 few-shot 与候选 few-shot 合并，保留最近最多 10 条。"""

        return (base_fewshots + candidate_fewshots)[-10:]

    def run_iteration(
        self,
        iteration: int,
        optimize_examples: list[BfclExample],
        regression_examples: list[BfclExample],
    ) -> IterationReport:
        """运行单次迭代：优化评估、反思、候选生成、回归门控、接受或回滚。"""

        self._emit_progress(event="iteration_start", iteration=iteration, stage="iteration", current=0, total=1)
        current_skill = self.skill_store.read_skill()
        current_version = self.skill_store.snapshot(current_skill)
        base_fewshots = self.skill_store.read_fewshots(limit=5)

        # 1. 在优化集上评估当前 Skill
        metrics_before, bad_cases, _, optimize_records, optimize_scores = self.evaluate_examples(
            optimize_examples,
            current_skill,
            fewshots=base_fewshots,
            stage="optimize_eval",
            iteration=iteration,
            save_trace=False,
        )

        # 2. 挖掘经验、反思失败案例并生成候选 Skill
        self._emit_progress(event="stage", iteration=iteration, stage="reflect_and_propose", current=1, total=1)
        success_cases = self.experience_miner.success_cases(optimize_records, optimize_scores)
        diagnoses = self.reflector.diagnose(bad_cases, success_cases=success_cases)
        scoped_rules = self.experience_miner.propose_scoped_rules(bad_cases, diagnoses)
        candidate_fewshots = self.experience_miner.build_candidate_fewshots(bad_cases, success_cases)

        # 保存优化阶段轨迹和经验产物
        _, optimize_scores = self.evaluator.evaluate(optimize_records)
        optimize_trace = self._save_stage_trace(
            iteration, "optimize_eval", optimize_records, optimize_scores, metrics_before, bad_cases, diagnoses
        )
        experience_path = self._save_experience(
            iteration, success_cases, bad_cases, scoped_rules, candidate_fewshots, diagnoses
        )

        candidate_skill, new_rules = self.evolver.build_candidate_skill(current_skill, diagnoses, scoped_rules)
        candidate_fewshots_for_eval = self._build_candidate_fewshots(candidate_fewshots, base_fewshots)
        skill_diff = self._skill_diff(current_skill, candidate_skill)
        infra_summary = self._infra_failure_summary(diagnoses, metrics_before["total"])

        gate_details: dict[str, Any] = {
            "min_improvement": self.min_improvement,
            "max_regression_drop": self.max_regression_drop,
            "new_rules": new_rules,
            "scoped_rules": scoped_rules,
            "candidate_fewshot_count": len(candidate_fewshots),
            "optimize_trace": optimize_trace,
            "experience_path": experience_path,
            "diagnosis_distribution": self._count_by_key(diagnoses, "skill_patch_type"),
            "bad_case_count": len(bad_cases),
            "success_case_count": len(success_cases),
            "candidate_skill_diff": skill_diff,
            **infra_summary,
        }

        # 基础设施失败占比过高时直接拒绝，避免在不可靠的运行时上继续迭代
        if infra_summary["infra_failure_rate"] >= 0.5:
            gate_details["decision"] = "rejected_infrastructure_failures"
            report = IterationReport(
                iteration=iteration,
                accepted=False,
                current_skill_version=current_version,
                candidate_skill_version=None,
                metrics_before=metrics_before,
                metrics_after=None,
                bad_cases=bad_cases,
                diagnoses=diagnoses,
                gate_reason="Rejected: model/API failures dominate this iteration. Fix runtime first.",
                gate_details=gate_details,
            )
            self._save_report(report)
            self._emit_progress(event="iteration_end", iteration=iteration, stage="iteration", accepted=False)
            return report

        # 没有可复用规则或 few-shot 候选时也拒绝
        if not new_rules and not candidate_fewshots:
            gate_details["decision"] = "rejected_no_candidate_memory"
            report = IterationReport(
                iteration=iteration,
                accepted=False,
                current_skill_version=current_version,
                candidate_skill_version=None,
                metrics_before=metrics_before,
                metrics_after=None,
                bad_cases=bad_cases,
                diagnoses=diagnoses,
                gate_reason="No reusable Skill rule or few-shot candidate was proposed.",
                gate_details=gate_details,
            )
            self._save_report(report)
            self._emit_progress(event="iteration_end", iteration=iteration, stage="iteration", accepted=False)
            return report

        # 3. 在回归集上对比当前 Skill 与候选 Skill
        regression_before, _, regression_before_trace, before_records, before_scores = self.evaluate_examples(
            regression_examples,
            current_skill,
            fewshots=base_fewshots,
            stage="regression_baseline",
            iteration=iteration,
        )
        regression_after, _, regression_after_trace, after_records, after_scores = self.evaluate_examples(
            regression_examples,
            candidate_skill,
            fewshots=candidate_fewshots_for_eval,
            stage="regression_candidate",
            iteration=iteration,
        )

        regression_diff = self._compare_regression(before_records, before_scores, after_records, after_scores)
        before_score = regression_before["exact_match_rate"]
        after_score = regression_after["exact_match_rate"]
        improvement = after_score - before_score
        regression_drop = max(0.0, before_score - after_score)
        candidate_version = self.skill_store.snapshot(candidate_skill)

        # 更新 champion 记录
        if self.champion_exact_match is None:
            self.champion_exact_match = before_score
            self.champion_skill_version = current_version
        elif before_score > self.champion_exact_match:
            self.champion_exact_match = before_score
            self.champion_skill_version = current_version

        champion_before = self.champion_exact_match
        champion_improvement = after_score - champion_before
        accepted = (
            improvement >= self.min_improvement
            and regression_drop <= self.max_regression_drop
            and champion_improvement >= self.min_improvement
        )

        # 4. 根据门控结果决定接受候选更新或回滚
        if accepted:
            if new_rules:
                self.skill_store.write_skill(candidate_skill)
            for item in candidate_fewshots:
                self.skill_store.append_fewshot(item)
            self.champion_exact_match = after_score
            self.champion_skill_version = candidate_version
            decision = "accepted"
            gate_reason = (
                f"Accepted: regression exact_match {before_score:.4f} -> {after_score:.4f} "
                f"(delta={improvement:.4f}, champion_delta={champion_improvement:.4f}, "
                f"improved={regression_diff['improved_count']}, "
                f"regressed={regression_diff['regressed_count']})."
            )
        elif regression_drop > self.max_regression_drop:
            decision = "rejected_regression_drop"
            gate_reason = (
                f"Rejected: candidate dropped held-out exact_match from {before_score:.4f} to {after_score:.4f} "
                f"(drop={regression_drop:.4f}, threshold={self.max_regression_drop:.4f})."
            )
        elif champion_improvement < self.min_improvement:
            decision = "rejected_below_champion"
            gate_reason = (
                f"Rejected: candidate exact_match {after_score:.4f} does not beat champion "
                f"{champion_before:.4f} by threshold {self.min_improvement:.4f}."
            )
        else:
            decision = "rejected_low_improvement"
            gate_reason = (
                f"Rejected: improvement {improvement:.4f} below threshold {self.min_improvement:.4f} "
                f"(improved={regression_diff['improved_count']}, regressed={regression_diff['regressed_count']})."
            )

        gate_details.update(
            {
                "decision": decision,
                "regression_before": regression_before,
                "regression_after": regression_after,
                "regression_before_trace": regression_before_trace,
                "regression_after_trace": regression_after_trace,
                "before_exact_match_rate": before_score,
                "after_exact_match_rate": after_score,
                "round_deployed_exact_match_rate": after_score if accepted else before_score,
                "deployed_exact_match_rate": self.champion_exact_match,
                "champion_before_exact_match_rate": champion_before,
                "champion_after_exact_match_rate": self.champion_exact_match,
                "champion_skill_version": self.champion_skill_version,
                "champion_improvement": round(champion_improvement, 4),
                "improvement": round(improvement, 4),
                "regression_drop": round(regression_drop, 4),
                "regression_diff": regression_diff,
            }
        )

        report = IterationReport(
            iteration=iteration,
            accepted=accepted,
            current_skill_version=current_version,
            candidate_skill_version=candidate_version,
            metrics_before=metrics_before,
            metrics_after=regression_after,
            bad_cases=bad_cases,
            diagnoses=diagnoses,
            gate_reason=gate_reason,
            gate_details=gate_details,
        )
        self._save_report(report)
        self._emit_progress(event="iteration_end", iteration=iteration, stage="iteration", accepted=accepted)
        return report

    def run(
        self,
        optimize_examples: list[BfclExample],
        regression_examples: list[BfclExample],
        iterations: int,
    ) -> list[IterationReport]:
        """运行多轮迭代，并保存汇总报告。"""

        reports: list[IterationReport] = []
        for iteration in range(iterations):
            reports.append(self.run_iteration(iteration, optimize_examples, regression_examples))
        self._save_summary(reports)
        return reports

    def _save_report(self, report: IterationReport) -> None:
        """将单次迭代的报告保存为 JSON。"""

        path = self.run_dir / f"iteration_{report.iteration:02d}.json"
        path.write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_summary(self, reports: list[IterationReport]) -> None:
        """将所有迭代报告汇总保存为 summary.json。"""

        path = self.run_dir / "summary.json"
        payload = [report.__dict__ for report in reports]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _count_by_key(items: list[dict], key: str) -> dict[str, int]:
        """按指定键统计列表中各值出现的次数。"""

        counts: dict[str, int] = {}
        for item in items:
            value = str(item.get(key, "unknown"))
            counts[value] = counts.get(value, 0) + 1
        return counts
