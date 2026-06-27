"""对 Agent 的执行轨迹进行成功经验挖掘。

本模块将原始执行轨迹转化为可复用的候选 skill。

1. 从精确匹配轨迹中保留正例。
2. 从 Bad Cases 中保留负例。
3. 对比正负两侧，提出有作用域的规则 (Scope rules)。
4. 构建候选 few-shot 记忆，通过 Skill 进行回归测试后进行持久化。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .schema import CaseScore, PredictionRecord


def _calls_to_json(calls: list) -> list[dict]:
    return [{"name": call.name, "arguments": call.arguments} for call in calls]


def _gold_value_candidates(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _contains_upper_token(values: list[Any]) -> bool:
    return any(isinstance(value, str) and value.isupper() and len(value) in {2, 3} for value in values)


def _contains_number(values: list[Any]) -> bool:
    return any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)


class ExperienceMiner:
    """从一次已评估的 Agent 轨迹中挖掘正例与负例经验。"""

    def success_cases(self, records: list[PredictionRecord], scores: list[CaseScore], limit: int = 8) -> list[dict]:
        """从精确匹配样例中返回精简的正例轨迹。"""

        score_by_id = {score.example_id: score for score in scores}
        cases: list[dict] = []
        for record in records:
            score = score_by_id[record.example.id]
            if not score.exact_match:
                continue
            cases.append(
                {
                    "id": record.example.id,
                    "user_query": record.example.user_query,
                    "tools": [tool.name for tool in record.example.tools],
                    "gold_calls": _calls_to_json(record.example.gold_calls),
                    "predicted_calls": _calls_to_json(record.predicted_calls),
                    "note": "Positive trace: the model matched the BFCL gold call exactly.",
                }
            )
            if len(cases) >= limit:
                break
        return cases

    def build_candidate_fewshots(
        self,
        bad_cases: list[dict],
        success_cases: list[dict],
        limit: int = 6,
    ) -> list[dict]:
        """构建一个小型对比式 few-shot 集合，用于候选评估。"""

        items: list[dict] = []
        for case in success_cases[: max(1, limit // 2)]:
            items.append(
                {
                    "user_query": case["user_query"],
                    "calls": case["gold_calls"],
                    "note": "Accepted positive trace from the current run.",
                }
            )

        remaining = max(0, limit - len(items))
        for case in bad_cases[:remaining]:
            parse_error = case.get("parse_error") or ""
            if parse_error.startswith("model_call_error:"):
                continue
            items.append(
                {
                    "user_query": case["user_query"],
                    "calls": case["gold_calls"],
                    "note": "Corrected negative trace from a Bad Case.",
                }
            )
        return items

    def propose_scoped_rules(self, bad_cases: list[dict], diagnoses: list[dict]) -> list[dict]:
        """从观察到的失败中推断窄范围规则。

        核心原则是避免模糊的全局规则，例如“精确提取参数”。
        有作用域的规则需附带证据，并且只有在 Harness 回归门禁确认其有效时才应被采纳。
        """

        diagnosis_by_id = {item["case_id"]: item for item in diagnoses}
        candidates: list[dict] = []
        evidence: dict[str, list[str]] = {}

        for case in bad_cases:
            diagnosis = diagnosis_by_id.get(case["id"], {})
            patch_type = diagnosis.get("skill_patch_type")
            if patch_type in {"infrastructure_error"}:
                continue

            if patch_type == "bfcl_exact_tool_name":
                rule_id = "bfcl_exact_tool_name"
                evidence.setdefault(rule_id, []).append(case["id"])
                candidates.append(
                    {
                        "rule_id": rule_id,
                        "scope": {"task_family": "bfcl", "applies_to": "tool_name"},
                        "rule": "Use one of the tool names exactly as provided in Available tools; never invent or rename a tool.",
                    }
                )

            if "argument_mismatch" not in case.get("error_types", []):
                continue
            if len(case.get("gold_calls", [])) != 1 or len(case.get("predicted_calls", [])) != 1:
                continue

            gold = case["gold_calls"][0]
            pred = case["predicted_calls"][0]
            gold_args = gold.get("arguments", {})
            pred_args = pred.get("arguments", {})

            extra_keys = sorted(set(pred_args) - set(gold_args))
            missing_keys = sorted(set(gold_args) - set(pred_args))
            if extra_keys or missing_keys:
                rule_id = "bfcl_schema_argument_names"
                evidence.setdefault(rule_id, []).append(case["id"])
                candidates.append(
                    {
                        "rule_id": rule_id,
                        "scope": {"task_family": "bfcl", "applies_to": "argument_names"},
                        "rule": "Use only argument names from the selected tool schema. Do not rename parameters or add schema-unknown keys.",
                    }
                )

            for arg_name, pred_value in pred_args.items():
                if arg_name not in gold_args:
                    continue
                gold_values = _gold_value_candidates(gold_args[arg_name])

                if isinstance(pred_value, str) and _contains_number(gold_values):
                    rule_id = "bfcl_numeric_json_types"
                    evidence.setdefault(rule_id, []).append(case["id"])
                    candidates.append(
                        {
                            "rule_id": rule_id,
                            "scope": {"argument_name": arg_name, "applies_to": "json_type"},
                            "rule": "When the schema or accepted answers imply a numeric argument, output a JSON number instead of a quoted string with units or symbols.",
                        }
                    )

                if "currency" in arg_name.lower() and _contains_upper_token(gold_values):
                    rule_id = "bfcl_currency_iso_codes"
                    evidence.setdefault(rule_id, []).append(case["id"])
                    candidates.append(
                        {
                            "rule_id": rule_id,
                            "scope": {"argument_name": arg_name, "applies_to": "currency_values"},
                            "rule": "For currency arguments, prefer uppercase ISO-style codes such as USD, EUR, and CNY when they are accepted by the task.",
                        }
                    )

                if isinstance(pred_value, str) and pred_value.startswith("$") and _contains_number(gold_values):
                    rule_id = "bfcl_strip_numeric_symbols"
                    evidence.setdefault(rule_id, []).append(case["id"])
                    candidates.append(
                        {
                            "rule_id": rule_id,
                            "scope": {"argument_name": arg_name, "applies_to": "numeric_values"},
                            "rule": "Strip currency symbols and unit text from numeric arguments; output the numeric value only.",
                        }
                    )

        # 按 rule_id 去重，保留首次出现的候选规则
        deduped: dict[str, dict] = {}
        for candidate in candidates:
            rule_id = candidate["rule_id"]
            if rule_id not in deduped:
                deduped[rule_id] = candidate

        # 统计每条规则背后的支撑案例数
        support = Counter(case_id for ids in evidence.values() for case_id in ids)
        del support
        results: list[dict] = []
        for rule_id, candidate in deduped.items():
            case_ids = sorted(set(evidence.get(rule_id, [])))
            candidate["support_count"] = len(case_ids)
            candidate["evidence_case_ids"] = case_ids[:8]
            results.append(candidate)

        # 按支撑数量降序、rule_id 升序排序
        return sorted(results, key=lambda item: (-item["support_count"], item["rule_id"]))
