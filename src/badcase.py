"""从已评估的轨迹中挖掘 Bad Case。"""

from __future__ import annotations

from .schema import CaseScore, PredictionRecord


class BadCaseMiner:
    """收集失败的样例及其上下文，以供后续反思使用。"""

    def mine(self, records: list[PredictionRecord], scores: list[CaseScore]) -> list[dict]:
        """从预测记录与评分中筛选非精确匹配的 Bad Case。"""

        score_by_id = {score.example_id: score for score in scores}
        bad_cases: list[dict] = []
        for record in records:
            score = score_by_id[record.example.id]
            # 精确匹配的样例不需要进入 Bad Case 集合
            if score.exact_match:
                continue
            bad_cases.append(
                {
                    "id": record.example.id,
                    "user_query": record.example.user_query,
                    "tags": record.example.tags,
                    "gold_calls": [call.__dict__ for call in record.example.gold_calls],
                    "predicted_calls": [call.__dict__ for call in record.predicted_calls],
                    "raw_output": record.raw_output,
                    "error_types": score.error_types,
                    "parse_error": record.parse_error,
                }
            )
        return bad_cases
