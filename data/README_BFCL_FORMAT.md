# BFCL-Compatible Data Format

本 Demo 使用 BFCL 风格的 JSONL 格式，每一行代表一个函数调用任务。

字段说明：

```json
{
  "id": "case_001",
  "user_query": "用户自然语言请求",
  "tools": [
    {
      "name": "工具名",
      "description": "工具说明",
      "parameters": {
        "字段名": "字段说明"
      }
    }
  ],
  "gold_calls": [
    {
      "name": "应该调用的工具名",
      "arguments": {
        "参数名": "参数值"
      }
    }
  ],
  "tags": ["single_tool", "argument_extraction"]
}
```

本格式保留了 BFCL 的核心思想：

- 给定用户请求。
- 给定候选工具列表。
- 模型输出函数调用。
- Evaluator 比对函数名、参数、顺序和格式。

如果使用官方 BFCL 数据，可以写一个转换脚本，把官方字段映射成：

- `user_query`
- `tools`
- `gold_calls`
- `tags`

然后复用本 Demo 的 Harness。
