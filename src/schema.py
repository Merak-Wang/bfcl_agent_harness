"""Agent Harness 的数据结构。

在核心流水线中传递的对象使用数据类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """模型的可调用工具定义。"""

    name: str
    description: str
    parameters: dict[str, str]


@dataclass(frozen=True)
class FunctionCall:
    """Agent 预测和标准答案使用的结构化函数调用的数据类。"""

    name: str
    arguments: JsonDict


@dataclass(frozen=True)
class BfclExample:
    """BFCL 的工具调用任务。"""

    id: str
    user_query: str
    tools: list[ToolSpec]
    gold_calls: list[FunctionCall]
    tags: list[str] = field(default_factory=list)


@dataclass
class PredictionRecord:
    """单个样例的完整执行记录。"""

    example: BfclExample
    predicted_calls: list[FunctionCall]
    raw_output: str
    parse_error: str | None = None
    latency_seconds: float = 0.0
    prompt_chars: int = 0
    raw_output_chars: int = 0


@dataclass
class CaseScore:
    """单个样例的细粒度评分。

    BFCL 评估通常关注函数名、参数和调用顺序。
    """

    example_id: str
    exact_match: bool
    function_match: bool
    argument_match: bool
    order_match: bool
    json_valid: bool
    error_types: list[str]


@dataclass
class IterationReport:
    """Harness 进行一次 Agent 迭代产生的指标与结果。"""

    iteration: int
    accepted: bool
    current_skill_version: str
    candidate_skill_version: str | None
    metrics_before: JsonDict
    metrics_after: JsonDict | None
    bad_cases: list[JsonDict]
    diagnoses: list[JsonDict]
    gate_reason: str
    gate_details: JsonDict = field(default_factory=dict)
