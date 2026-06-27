"""用于加载兼容 BFCL 的 JSONL 数据集工具。"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import BfclExample, FunctionCall, ToolSpec


def load_bfcl_jsonl(path: str | Path) -> list[BfclExample]:
    """从 JSONL 文件中加载样例。"""

    examples: list[BfclExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            try:
                tools = [ToolSpec(**tool) for tool in item["tools"]]
                gold_calls = [FunctionCall(**call) for call in item["gold_calls"]]
                examples.append(
                    BfclExample(
                        id=item["id"],
                        user_query=item["user_query"],
                        tools=tools,
                        gold_calls=gold_calls,
                        tags=item.get("tags", []),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing field {exc} at line {line_no}") from exc
    return examples


def split_examples(
    examples: list[BfclExample],
    validation_ratio: float = 0.7,
) -> tuple[list[BfclExample], list[BfclExample]]:
    """将数据集划分为优化集与回归集。

    应固定回归集（但两个集合的分布可能不同）。Evolver 只能查看优化集中的 Bad Cases，回归门会在回归集上验证，避免过拟合。
    """

    if not examples:
        return [], []
    cut = max(1, int(len(examples) * validation_ratio))
    return examples[:cut], examples[cut:]
