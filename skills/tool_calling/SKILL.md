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

### Auto-Evolution Patch (2026-06-26 07:08:05 UTC)
- [bfcl_numeric_json_types] When the schema or accepted answers imply a numeric argument, output a JSON number instead of a quoted string with units or symbols. Scope: {'argument_name': 'mass', 'applies_to': 'json_type'}. Support: 4. Evidence: multiple_20, simple_45, simple_61, simple_69.
- [bfcl_schema_argument_names] Use only argument names from the selected tool schema. Do not rename parameters or add schema-unknown keys. Scope: {'task_family': 'bfcl', 'applies_to': 'argument_names'}. Support: 3. Evidence: multiple_20, simple_14, simple_58.
- [bfcl_exact_tool_name] Use one of the tool names exactly as provided in Available tools; never invent or rename a tool. Scope: {'task_family': 'bfcl', 'applies_to': 'tool_name'}. Support: 1. Evidence: multiple_7.
- [llm_reflector_rule] Strictly respect the JSON data type of each argument as defined in the function schema. If the schema expects a string, provide a quoted string (e.g., '2022'); if it expects a number, provide an unquoted number (e.g., 2022). Do not add extra characters like units or commas. Scope: {'task_family': 'bfcl', 'applies_to': 'json_type'}. Rationale: Model provided a string with units or quotes when a number was expected, or a number when a string was expected, violating the schema data type.
- [llm_reflector_rule] For arguments that are arrays or objects, use the exact JSON structure described in the parameter schema (e.g., a list of strings, an object with specific keys). Do not flatten them into a single string. Scope: {'task_family': 'bfcl', 'applies_to': 'format'}. Rationale: Model passed a flat string when an array or object structure was required, e.g., 'GC' instead of [['G','C'],['C','G']].

### Auto-Evolution Patch (2026-06-26 07:12:23 UTC)
- [bfcl_schema_argument_names] Use only argument names from the selected tool schema. Do not rename parameters or add schema-unknown keys. Scope: {'task_family': 'bfcl', 'applies_to': 'argument_names'}. Support: 2. Evidence: simple_14, simple_58.
- [llm_reflector_rule] When a tool parameter expects a specific value from a set (e.g., substance, species, time_frame), provide exactly that value, not a synonym or paraphrase. Scope: {'task_family': 'bfcl', 'applies_to': 'format'}. Rationale: Used paraphrased or incorrect value for parameter (e.g., 'water' instead of 'ice', 'human' instead of 'Homo sapiens', 'last six months' instead of 'six_months').
- [llm_reflector_rule] When a tool parameter expects a structured JSON type (object, array, array of arrays), produce the argument with that exact structure, ensuring proper nesting and primitive types (number, string). Scope: {'task_family': 'bfcl', 'applies_to': 'json_type'}. Rationale: Provided argument in wrong data type (scalar instead of list, string instead of object, nested list instead of flat string).
- [llm_reflector_rule] When providing a mathematical function as a string, use Python's ** operator for exponents and include spaces around operators; avoid superfluous multiplication signs between coefficients and variables. Scope: {'task_family': 'bfcl', 'applies_to': 'format'}. Rationale: Used incorrect mathematical notation (e.g., '*' for multiplication, missing spaces) that does not match any acceptable form.
- [llm_reflector_rule] If a tool provides a boolean parameter indicating specific details (e.g., 'specific_function', 'detailed'), set it to true when the user query explicitly asks for that specific information. Scope: {'task_family': 'bfcl', 'applies_to': 'argument_name'}. Rationale: Omitted a boolean parameter (e.g., 'specific_function') that is required to fulfill the user's request for specific details.
