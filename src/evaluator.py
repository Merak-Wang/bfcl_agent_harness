"""自动评估器。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .schema import CaseScore, FunctionCall, PredictionRecord


def _normalize(value: Any) -> Any:
    """对值进行轻微归一化，以实现更鲁棒的比较。"""

    if isinstance(value, str):
        return value.strip()
    return value


def _value_matches(pred_value: Any, gold_value: Any) -> bool:
    """判断预测值是否匹配 BFCL 风格的标准答案。

    官方 BFCL 的 possible-answer 文件将每个参数值存为一个可接受候选列表，例如：

        {"unit": ["units", ""]}

    只要模型预测了列表中的任意一个值，就应视为正确。
    如果标准值不是列表，则退回到归一化后的精确相等比较。
    """

    normalized_pred = _normalize(pred_value)
    if isinstance(gold_value, list):
        return any(normalized_pred == _normalize(candidate) for candidate in gold_value)
    return normalized_pred == _normalize(gold_value)


def _missing_is_allowed(gold_value: Any) -> bool:
    """BFCL 通常用空字符串候选表示可选参数。"""

    return isinstance(gold_value, list) and any(candidate == "" for candidate in gold_value)


def _arguments_equal(pred: dict, gold: dict) -> bool:
    """比较预测参数与标准参数是否一致，允许可选参数缺失。"""

    extra_keys = set(pred.keys()) - set(gold.keys())
    if extra_keys:
        return False

    for key, gold_value in gold.items():
        if key not in pred:
            if _missing_is_allowed(gold_value):
                continue
            return False
        if not _value_matches(pred[key], gold_value):
            return False
    return True


class BfclEvaluator:
    """评估函数调用、参数、调用顺序以及 JSON 有效性的精确匹配情况。"""

    def score_case(self, record: PredictionRecord) -> CaseScore:
        """为单个预测记录计算细粒度评分。"""

        gold = record.example.gold_calls
        pred = record.predicted_calls
        json_valid = record.parse_error is None

        # 函数名匹配：预测调用数量与名称都需与标准一致
        function_match = len(pred) == len(gold) and all(p.name == g.name for p, g in zip(pred, gold))
        # 参数匹配：每个对应调用的参数需一致
        argument_match = (
            len(pred) == len(gold) and all(_arguments_equal(p.arguments, g.arguments) for p, g in zip(pred, gold))
        )
        # 顺序匹配：调用顺序的函数名序列需完全一致
        order_match = [p.name for p in pred] == [g.name for g in gold]
        exact_match = json_valid and function_match and argument_match and order_match

        # 收集错误类型，便于后续诊断
        error_types: list[str] = []
        if not json_valid:
            error_types.append("json_invalid")
        if len(pred) != len(gold):
            error_types.append("call_count_mismatch")
        if not function_match:
            error_types.append("function_mismatch")
        if not argument_match:
            error_types.append("argument_mismatch")
        if not order_match:
            error_types.append("order_mismatch")

        return CaseScore(
            example_id=record.example.id,
            exact_match=exact_match,
            function_match=function_match,
            argument_match=argument_match,
            order_match=order_match,
            json_valid=json_valid,
            error_types=error_types,
        )

    def evaluate(self, records: list[PredictionRecord]) -> tuple[dict, list[CaseScore]]:
        """评估一批记录，返回汇总指标与每条记录的评分。"""

        scores = [self.score_case(record) for record in records]
        total = max(1, len(scores))
        error_counter = Counter(error for score in scores for error in score.error_types)

        total_latency = sum(record.latency_seconds for record in records)
        total_prompt_chars = sum(record.prompt_chars for record in records)
        total_raw_chars = sum(record.raw_output_chars for record in records)

        metrics = {
            "total": len(scores),
            "exact_match_rate": round(sum(s.exact_match for s in scores) / total, 4),
            "function_match_rate": round(sum(s.function_match for s in scores) / total, 4),
            "argument_match_rate": round(sum(s.argument_match for s in scores) / total, 4),
            "order_match_rate": round(sum(s.order_match for s in scores) / total, 4),
            "json_valid_rate": round(sum(s.json_valid for s in scores) / total, 4),
            "error_distribution": dict(error_counter),
            "total_latency_seconds": round(total_latency, 4),
            "avg_latency_seconds": round(total_latency / total, 4),
            "total_prompt_chars": total_prompt_chars,
            "avg_prompt_chars": round(total_prompt_chars / total, 2),
            "total_raw_output_chars": total_raw_chars,
            "avg_raw_output_chars": round(total_raw_chars / total, 2),
        }
        return metrics, scores
