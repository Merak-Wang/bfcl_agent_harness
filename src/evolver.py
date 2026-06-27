"""进化模块。

根据测试/诊断结果，自动给 SKILL.md 文件追加新的经验规则，让 agent 下次遇到类似问题能做得更好。
进化器为每次迭代创建一个候选的更新。
测试框架在回归数据集上评估候选，仅在提升指标时才保留它。
"""

from __future__ import annotations

from datetime import datetime, timezone


class SkillEvolver:
    """从反思结果中生成候选 skill 的内容。"""

    PATCH_RULES = {
        "pure_json_no_markdown": (
            "Return only a raw JSON array of tool calls. Do not wrap the answer in Markdown fences such as ```json."
        ),
        "bfcl_exact_tool_name": (
            "For BFCL tasks, use function names exactly as provided in the available tools. Do not rename, translate, abbreviate, or invent tools."
        ),
        "bfcl_argument_precision": (
            "For BFCL tasks, extract argument values exactly from the user query. Omit optional arguments unless the user explicitly provides them."
        ),
        "python_formula_syntax": (
            "When an argument expects a mathematical formula or Python expression, use Python-style syntax such as x**2 instead of x^2."
        ),
        "prefer_calculate_discount": (
            "Prefer calculate_discount over calculator when the query asks for percent-off, discount, or final price after discount."
        ),
        "extract_discount_price": (
            "Extract original price from phrasing such as 'priced at 299' or 'price of 299'; do not use the discount percent as the price."
        ),
        "resolve_contact_dependency": (
            "Resolve contact before message or meeting when the user gives a person name and the target tool requires an email address."
        ),
        "respect_negation": (
            "Negation should block destructive tools. For example, if the user says do not cancel, call get_order_status instead of cancel_order."
        ),
        "normalize_cancel_reason": (
            "Normalize cancellation reasons by removing leading first-person pronouns such as 'I' before filling the reason argument."
        ),
    }

    def build_candidate_skill(
        self,
        current_skill: str,
        diagnoses: list[dict],
        scoped_rules: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        """ 将一组紧凑的可重用规则附加到SKILL.md。
        
        该方法支持规则模式和 LLM 模式。
        """

        selected_rules: list[str] = []
        seen = current_skill.lower()
        scoped_rules = scoped_rules or []

        for rule_item in scoped_rules:
            support = rule_item.get("support_count", 0)
            if support <= 0:
                continue
            rule_text = self._format_scoped_rule(rule_item)
            if rule_text.lower() not in seen and rule_text not in selected_rules:
                selected_rules.append(rule_text)

        for diagnosis in diagnoses:
            patch_type = diagnosis.get("skill_patch_type")
            if patch_type in {"infrastructure_error", "reflection_infrastructure_error"}:
                continue

            llm_rule = str(diagnosis.get("reusable_rule", "")).strip()
            if patch_type == "llm_reusable_rule" and llm_rule:
                rule = self._format_llm_rule(diagnosis)
            elif patch_type in {"bfcl_argument_precision"} and scoped_rules:
                # Prefer scoped rules over vague global precision rules.
                continue
            else:
                rule = self.PATCH_RULES.get(patch_type)

            if rule and rule.lower() not in seen and rule not in selected_rules:
                selected_rules.append(rule)

        if not selected_rules:
            return current_skill, []

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        block = ["", f"### Auto-Evolution Patch ({timestamp})"]
        for rule in selected_rules[:5]:
            block.append(f"- {rule}")
        block.append("")
        candidate = current_skill.rstrip() + "\n" + "\n".join(block)
        return candidate, selected_rules[:5]

    @staticmethod
    def _format_scoped_rule(rule_item: dict) -> str:
        """把一条数据挖掘出的 scoped rule 格式化成紧凑文本。"""

        scope = rule_item.get("scope", {})
        evidence = ", ".join(rule_item.get("evidence_case_ids", [])[:4])
        evidence_text = f" Evidence: {evidence}." if evidence else ""
        return (
            f"[{rule_item.get('rule_id', 'scoped_rule')}] {rule_item['rule']} "
            f"Scope: {scope}. Support: {rule_item.get('support_count', 0)}.{evidence_text}"
        )

    @staticmethod
    def _format_llm_rule(diagnosis: dict) -> str:
        """把 LLM 反思生成的可重用规则格式化成文本。"""

        rule = str(diagnosis.get("reusable_rule", "")).strip()
        scope = diagnosis.get("scope", {})
        root = str(diagnosis.get("root_cause", "")).strip()
        prefix = "[llm_reflector_rule]"
        scope_text = f" Scope: {scope}." if scope else ""
        root_text = f" Rationale: {root}" if root else ""
        return f"{prefix} {rule}{scope_text}{root_text}"

    def fewshot_items_from_bad_cases(self, bad_cases: list[dict], limit: int = 3) -> list[dict]:
        """把失败案例 Bad Cases 转成 few-shot 示例。"""

        items: list[dict] = []
        for case in bad_cases[:limit]:
            items.append(
                {
                    "user_query": case["user_query"],
                    "calls": case["gold_calls"],
                    "note": "Added automatically from a verified Bad Case.",
                }
            )
        return items

    def fewshot_items_from_success_cases(self, success_cases: list[dict], limit: int = 3) -> list[dict]:
        """把成功案例 (positive traces) 转成 few-shot 示例，用于巩固已掌握的能力。"""

        items: list[dict] = []
        for case in success_cases[:limit]:
            items.append(
                {
                    "user_query": case["user_query"],
                    "calls": case["gold_calls"],
                    "note": "Added automatically from a verified positive trace.",
                }
            )
        return items
