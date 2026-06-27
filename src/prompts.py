"""Prompt 组装。

离线规则模型不依赖此 prompt，但调用真实 API 的模型需要它。
清晰的输出约束可以减少 json_invalid 类失败。
"""

from __future__ import annotations

import json

from .schema import BfclExample


def build_tool_calling_prompt(
    example: BfclExample,
    skill_text: str,
    fewshots: list[dict],
) -> str:
    """为单次工具调用任务构建完整的模型 prompt。"""

    # 将工具定义序列化为模型可识别的 JSON 描述
    tool_payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in example.tools
    ]

    return (
        "You are a function-calling agent.\n"
        "Use the following Skill as procedural memory.\n\n"
        f"{skill_text}\n\n"
        "Few-shot examples:\n"
        f"{json.dumps(fewshots, ensure_ascii=False, indent=2)}\n\n"
        "Available tools:\n"
        f"{json.dumps(tool_payload, ensure_ascii=False, indent=2)}\n\n"
        f"User query: {example.user_query}\n\n"
        "Output requirements:\n"
        "1. Return ONLY a raw JSON array. Do not wrap it in ```json fences.\n"
        '2. Each item must be {"name": "tool_name", "arguments": {...}}.\n'
        "3. Use tool names exactly as listed. Do not invent tools.\n"
        "4. For optional arguments, omit them unless the user explicitly provides the value.\n"
        "5. Use Python-style expressions when a function argument expects a formula, for example x**2 instead of x^2.\n"
    )
