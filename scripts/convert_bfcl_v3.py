"""将官方 BFCL v3 文件转换为本 demo 的 JSONL 格式。

官方 Hugging Face 数据集将题目与可能答案分开存储：

- BFCL_v3_simple(1).jsonl       -> 题目/工具
- BFCL_v3_simple.jsonl          -> 可能答案
- BFCL_v3_multiple(1).jsonl     -> 题目/工具
- BFCL_v3_multiple.jsonl        -> 可能答案

两者都是 JSONL 文件。本脚本按 `id` 将它们合并，并写入一个
可由 demo Harness 直接加载的单一文件。

示例：

    python scripts/convert_bfcl_v3.py \
      --questions raw_bfcl/BFCL_v3_simple_questions.jsonl raw_bfcl/BFCL_v3_multiple_questions.jsonl \
      --answers raw_bfcl/BFCL_v3_simple_answers.jsonl raw_bfcl/BFCL_v3_multiple_answers.jsonl \
      --output data/bfcl_v3_simple_multiple.jsonl \
      --limit-per-file 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，跳过空行。"""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows


def extract_user_query(question_field: Any) -> str:
    """从 BFCL 嵌套的 `question` 字段中提取第一条用户消息。"""

    # BFCL v3 通常使用 [[{"role": "user", "content": "..."}]] 这种嵌套结构。
    queue = [question_field]
    while queue:
        item = queue.pop(0)
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content", ""))
        if isinstance(item, list):
            queue = item + queue
    if isinstance(question_field, str):
        return question_field
    return json.dumps(question_field, ensure_ascii=False)


def normalize_tools(item: dict[str, Any]) -> list[dict[str, Any]]:
    """将 BFCL 函数模式转换为本 demo 的精简 ToolSpec 格式。"""

    raw_tools = item.get("function") or item.get("functions") or item.get("tools") or []
    tools: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        fn = raw_tool.get("function", raw_tool) if isinstance(raw_tool, dict) else {}
        params = fn.get("parameters", {})
        properties = params.get("properties", {}) if isinstance(params, dict) else {}
        compact_params = {
            name: prop.get("description", prop.get("type", "")) if isinstance(prop, dict) else str(prop)
            for name, prop in properties.items()
        }
        tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": compact_params,
            }
        )
    return tools


def normalize_gold_calls(answer_item: dict[str, Any]) -> list[dict[str, Any]]:
    """将 BFCL 的 `ground_truth` 转换为 demo 的 `gold_calls`。

    BFCL 将每个标准调用存储为：

        {"function_name": {"arg": [acceptable_value_1, acceptable_value_2]}}

    我们有意保留这些可接受值列表。评估器已扩展为：
    只要预测值落在列表中即视为正确。
    """

    calls: list[dict[str, Any]] = []
    for call in answer_item.get("ground_truth", []):
        if not isinstance(call, dict):
            continue
        for name, arguments in call.items():
            calls.append({"name": name, "arguments": arguments or {}})
    return calls


def category_from_id(example_id: str) -> str:
    """从样例 ID 中提取类别前缀。"""

    return example_id.split("_", 1)[0] if "_" in example_id else "bfcl"


def convert_pair(question_path: Path, answer_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """将一组题目文件与答案文件配对转换。"""

    questions = read_jsonl(question_path)
    answers = {row["id"]: row for row in read_jsonl(answer_path)}

    converted: list[dict[str, Any]] = []
    for item in questions:
        example_id = item["id"]
        answer = answers.get(example_id)
        if answer is None:
            continue
        converted.append(
            {
                "id": example_id,
                "user_query": extract_user_query(item.get("question", "")),
                "tools": normalize_tools(item),
                "gold_calls": normalize_gold_calls(answer),
                "tags": [category_from_id(example_id), "official_bfcl_v3"],
            }
        )
        if limit is not None and len(converted) >= limit:
            break
    return converted


def main() -> None:
    """命令行入口：解析参数并执行转换。"""

    parser = argparse.ArgumentParser(description="Convert official BFCL v3 question/answer JSONL to demo JSONL.")
    parser.add_argument("--questions", nargs="+", required=True, help="Question/tool files.")
    parser.add_argument("--answers", nargs="+", required=True, help="Matching possible-answer files.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit-per-file", type=int, default=None, help="Optional sample limit per input pair.")
    args = parser.parse_args()

    if len(args.questions) != len(args.answers):
        raise ValueError("--questions and --answers must contain the same number of files.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for question_path, answer_path in zip(args.questions, args.answers):
        rows = convert_pair(Path(question_path), Path(answer_path), args.limit_per_file)
        print(f"Converted {len(rows)} rows from {question_path}")
        all_rows.extend(rows)

    with output_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
