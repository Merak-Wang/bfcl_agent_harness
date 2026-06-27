# Tool Calling Skill

## Purpose

Use this skill when the task asks the agent to select one or more tools and produce structured function calls.

This file is the procedural memory of the Agent Harness. It should capture reusable rules, common failure patterns, and verified fixes. The large model parameters are frozen; improvements should happen by editing this skill, the prompt, or the few-shot examples.

## Output Contract

Always return a JSON array of function calls:

```json
[
  {
    "name": "tool_name",
    "arguments": {
      "arg_name": "arg_value"
    }
  }
]
```

Do not include natural language outside the JSON array.

Never wrap the JSON array in Markdown fences such as ```json.

## General Rules

1. Select tools by matching the user goal with the tool description.
2. Extract only arguments required by the selected tool.
3. Preserve exact entities such as names, cities, dates, order ids, and quoted text.
4. For multi-step tasks, output calls in dependency order.
5. If one tool needs an email address but the user only gives a person name, first call the contact lookup tool.
6. For BFCL tasks, use the function name exactly as listed in the available tools.
7. Omit optional arguments unless the user explicitly provides them.
8. For formula-like arguments, use Python-style expressions such as x**2 instead of x^2.

## Known Fixes

This section is intentionally short at initialization. The Harness may append verified fixes here after evaluating Bad Cases.
