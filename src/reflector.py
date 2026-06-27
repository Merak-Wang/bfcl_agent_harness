"""Failure reflection module.

The Reflector is the analysis component of the closed loop. It reads failed
trajectories and proposes explanations that may become candidate Skill updates.

Two implementations are provided:

- RuleFailureReflector: deterministic, cheap, and stable for offline demos.
- LLMFailureReflector: calls an OpenAI-compatible model once per iteration to
  analyze positive and negative traces. This is closer to a Hermes-like
  reflective loop because the model can write new reusable rules that are not
  hard-coded in Python.

The Harness still owns the safety gate. A Reflector only proposes; it never
persists Skill memory by itself.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol


class Reflector(Protocol):
    """Agent Harness使用的反射器。

    注入diagnose方法使用。
    """

    def diagnose(self, bad_cases: list[dict], success_cases: list[dict] | None = None) -> list[dict]:
        """依据错误案例和成功案例分析失败的轨迹并生成诊断。"""


class RuleFailureReflector:
    """固定的 Reflector.
    
    当LLM不可用或不需要时使用。它使用一组硬编码的规则来分析失败的轨迹并生成诊断。
    """

    def diagnose(self, bad_cases: list[dict], success_cases: list[dict] | None = None) -> list[dict]:
        del success_cases
        diagnoses: list[dict] = []
        for case in bad_cases:
            query = case["user_query"].lower()
            predicted_names = [call["name"] for call in case["predicted_calls"]]
            gold_names = [call["name"] for call in case["gold_calls"]]

            parse_error = case.get("parse_error") or ""
            if parse_error.startswith("model_call_error:"):
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "model_or_api_error",
                        "root_cause": parse_error,
                        "skill_patch_type": "infrastructure_error",
                        "fix": "Fix API key, base URL, model name, timeout, or rate limit before evolving Skill memory.",
                    }
                )
                continue

            if "json_invalid" in case["error_types"] and "```" in case.get("raw_output", ""):
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "markdown_json_fence",
                        "root_cause": "The model returned a JSON array wrapped in Markdown code fences.",
                        "skill_patch_type": "pure_json_no_markdown",
                        "fix": "Return a raw JSON array only. Never wrap tool calls in Markdown fences.",
                    }
                )
                continue

            if "official_bfcl_v3" in case.get("tags", []) and "function_mismatch" in case["error_types"]:
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "bfcl_tool_selection_error",
                        "root_cause": "The model selected a wrong or unavailable function name in an official BFCL task.",
                        "skill_patch_type": "bfcl_exact_tool_name",
                        "fix": "Use the function name exactly as listed in the provided tools and choose by description, not by memory.",
                    }
                )
                continue

            if "official_bfcl_v3" in case.get("tags", []) and "argument_mismatch" in case["error_types"]:
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "bfcl_argument_error",
                        "root_cause": "The model selected the right function family but missed BFCL-style argument extraction details.",
                        "skill_patch_type": "bfcl_argument_precision",
                        "fix": "Extract arguments from the user query exactly and omit optional arguments when not explicitly provided.",
                    }
                )
                continue

            if "^" in case.get("raw_output", "") and any("function" in call.get("arguments", {}) for call in case["gold_calls"]):
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "formula_format_error",
                        "root_cause": "The model used caret notation in a formula where Python-style exponentiation is expected.",
                        "skill_patch_type": "python_formula_syntax",
                        "fix": "Use Python-style formula syntax, for example x**2 instead of x^2.",
                    }
                )
                continue

            if "percent off" in query and "calculate_discount" in gold_names:
                patch_type = "prefer_calculate_discount"
                root_cause = "The agent used a generic calculator instead of the domain-specific discount tool."
                fix = "When the query asks for percent-off or discount price, prefer calculate_discount over calculator."
                if predicted_names == ["calculate_discount"] and "argument_mismatch" in case["error_types"]:
                    patch_type = "extract_discount_price"
                    root_cause = "The agent selected the correct discount tool but extracted the percent as the original price."
                    fix = "Extract original price from phrasing such as 'priced at 299', not from the percent value."
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "confusable_tool",
                        "root_cause": root_cause,
                        "skill_patch_type": patch_type,
                        "fix": fix,
                    }
                )
                continue

            if ("email" in query or "meeting" in query or "book" in query) and "get_contact" in gold_names:
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "missing_dependency_call",
                        "root_cause": "The agent used a person name directly where an email address was required.",
                        "skill_patch_type": "resolve_contact_dependency",
                        "fix": "Resolve contact before message or meeting when tools require an email address.",
                    }
                )
                continue

            if "do not cancel" in query and "cancel_order" in predicted_names:
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "negation_ignored",
                        "root_cause": "The agent ignored a negation instruction and chose a destructive tool.",
                        "skill_patch_type": "respect_negation",
                        "fix": "Negation should block destructive tools such as cancel_order.",
                    }
                )
                continue

            if "cancel order" in query and "because i " in query:
                diagnoses.append(
                    {
                        "case_id": case["id"],
                        "failure_type": "argument_normalization",
                        "root_cause": "The agent copied a first-person pronoun into a structured cancellation reason.",
                        "skill_patch_type": "normalize_cancel_reason",
                        "fix": "Normalize cancellation reasons by removing leading first-person pronouns such as 'I'.",
                    }
                )
                continue

            diagnoses.append(
                {
                    "case_id": case["id"],
                    "failure_type": "generic_tool_call_error",
                    "root_cause": "The prediction differs from the gold function call or arguments.",
                    "skill_patch_type": "add_fewshot",
                    "fix": "Add this case as a few-shot example for similar future queries.",
                }
            )
        return diagnoses


class LLMFailureReflector:
    """LLM-driven Reflector for trace analysis.

    LLM接收一小部分好的和坏的样本。它从中学习有用的规则和失败的经验，在写入skill之前经过回归验证，即可变成持久记忆。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 90,
        max_bad_cases: int = 16,
        max_success_cases: int = 8,
        fallback: Reflector | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("REFLECTOR_MODEL") or os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
        self.timeout = timeout
        self.max_bad_cases = max_bad_cases
        self.max_success_cases = max_success_cases
        self.fallback = fallback or RuleFailureReflector()

    def diagnose(self, bad_cases: list[dict], success_cases: list[dict] | None = None) -> list[dict]:
        if not bad_cases:
            return []
        if not self.api_key:
            return self.fallback.diagnose(bad_cases, success_cases)

        prompt = self._build_prompt(bad_cases[: self.max_bad_cases], (success_cases or [])[: self.max_success_cases])
        try:
            raw = self._chat(prompt)
            diagnoses = self._parse_diagnoses(raw)
            if not diagnoses:
                return self.fallback.diagnose(bad_cases, success_cases)
            return self._merge_with_fallback(bad_cases, success_cases, diagnoses)
        except Exception as exc:  # noqa: BLE001 - reflection must not break task evaluation.
            fallback = self.fallback.diagnose(bad_cases, success_cases)
            fallback.append(
                {
                    "case_id": "reflector",
                    "failure_type": "llm_reflector_error",
                    "root_cause": str(exc),
                    "skill_patch_type": "reflection_infrastructure_error",
                    "fix": "Check reflector API configuration. The deterministic fallback was used.",
                }
            )
            return fallback

    def _chat(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _build_prompt(self, bad_cases: list[dict], success_cases: list[dict]) -> str:
        compact_bad = [self._compact_case(item) for item in bad_cases]
        compact_success = [self._compact_success(item) for item in success_cases]
        return (
            "You are the Reflector of an Agent Harness.\n"
            "Analyze BFCL function-calling traces and propose reusable procedural memory.\n"
            "Use positive traces to understand what correct behavior looks like.\n"
            "Use bad traces to identify narrow, actionable failure causes.\n\n"
            "Return ONLY a raw JSON array. Each item must be:\n"
            "{\n"
            '  "case_id": "string or group id",\n'
            '  "failure_type": "short_name",\n'
            '  "root_cause": "specific explanation",\n'
            '  "skill_patch_type": "llm_reusable_rule|add_fewshot|infrastructure_error",\n'
            '  "fix": "short fix",\n'
            '  "reusable_rule": "one narrow rule suitable for SKILL.md",\n'
            '  "scope": {"task_family": "bfcl", "applies_to": "tool_name|argument_name|json_type|format"}\n'
            "}\n\n"
            "Rules for your analysis:\n"
            "- Prefer one or two high-quality reusable rules over many vague rules.\n"
            "- Do not propose a global rule like 'be precise'.\n"
            "- If a failure is very local, mark it add_fewshot instead of llm_reusable_rule.\n"
            "- If the issue is API timeout/key/rate limit, mark infrastructure_error.\n"
            "- Do not leak the gold answer into a rule unless it generalizes.\n\n"
            f"Positive traces:\n{json.dumps(compact_success, ensure_ascii=False, indent=2)}\n\n"
            f"Bad traces:\n{json.dumps(compact_bad, ensure_ascii=False, indent=2)}\n"
        )

    @staticmethod
    def _compact_case(case: dict) -> dict:
        return {
            "id": case.get("id"),
            "user_query": case.get("user_query"),
            "tags": case.get("tags", []),
            "gold_calls": case.get("gold_calls", []),
            "predicted_calls": case.get("predicted_calls", []),
            "error_types": case.get("error_types", []),
            "parse_error": case.get("parse_error"),
        }

    @staticmethod
    def _compact_success(case: dict) -> dict:
        return {
            "id": case.get("id"),
            "user_query": case.get("user_query"),
            "tools": case.get("tools", []),
            "gold_calls": case.get("gold_calls", []),
            "predicted_calls": case.get("predicted_calls", []),
        }

    @staticmethod
    def _extract_json_array(text: str) -> str:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate).strip()
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start != -1 and end != -1 and end > start:
            return candidate[start : end + 1]
        return candidate

    def _parse_diagnoses(self, raw: str) -> list[dict]:
        data = json.loads(self._extract_json_array(raw))
        if not isinstance(data, list):
            return []
        results: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id", "group"))
            patch_type = str(item.get("skill_patch_type", "llm_reusable_rule"))
            results.append(
                {
                    "case_id": case_id,
                    "failure_type": str(item.get("failure_type", "llm_trace_pattern")),
                    "root_cause": str(item.get("root_cause", "LLM reflector found a reusable trace pattern.")),
                    "skill_patch_type": patch_type,
                    "fix": str(item.get("fix", item.get("reusable_rule", "Use the reusable rule proposed by the LLM reflector."))),
                    "reusable_rule": str(item.get("reusable_rule", "")).strip(),
                    "scope": item.get("scope", {}) if isinstance(item.get("scope", {}), dict) else {},
                    "source": "llm_reflector",
                }
            )
        return results

    def _merge_with_fallback(
        self,
        bad_cases: list[dict],
        success_cases: list[dict] | None,
        llm_diagnoses: list[dict],
    ) -> list[dict]:
        """将 LLM 群体诊断与逐案例的后备诊断相结合。
        
        LLM通常返回少量的群体级模式。这对于技能进化是有益的，但报告仍应覆盖每个单独的坏案例。
        因此，我们保留LLM规则，并为LLM未明确提及的坏案例添加后备诊断。
        """

        mentioned = {str(item.get("case_id")) for item in llm_diagnoses}
        concrete_bad_case_ids = {str(item.get("id")) for item in bad_cases}
        covered = mentioned & concrete_bad_case_ids
        fallback_cases = [item for item in bad_cases if str(item.get("id")) not in covered]
        fallback = self.fallback.diagnose(fallback_cases, success_cases)
        return llm_diagnoses + fallback


# Backward-compatible name used by older imports.
FailureReflector = RuleFailureReflector
