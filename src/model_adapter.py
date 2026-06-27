"""模型适配器。

提供两种模型后端：

1. RuleBasedAdapter：离线且确定性的规则模型，适用于演示和调试。
2. OpenAICompatibleAdapter：调用任何兼容 /chat/completions 的模型 API。
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from .schema import BfclExample, FunctionCall


class ModelAdapter(ABC):
    """Harness 使用的抽象模型接口。"""

    @abstractmethod
    def generate_calls(
        self,
        example: BfclExample,
        prompt: str,
        skill_text: str,
        fewshots: list[dict],
    ) -> tuple[list[FunctionCall], str, str | None]:
        """返回解析后的调用、原始输出以及可选的解析错误。"""


def _json_calls_to_objects(raw: str) -> tuple[list[FunctionCall], str | None]:
    """将模型输出解析为 FunctionCall 对象。

    有些模型即使被要求返回原始 JSON，仍会返回 Markdown 包裹的 JSON：

        ```json
        [...]
        ```

    直接 json.loads(raw) 会把这些本可正确的输出标记为 json_invalid。
    因此解析器会先剥离 Markdown 围栏，再提取第一个 JSON 数组。
    """

    candidate = _extract_json_array(raw)

    try:
        data = json.loads(candidate)
        if not isinstance(data, list):
            return [], "Output is not a JSON array"
        calls = [FunctionCall(name=item["name"], arguments=item["arguments"]) for item in data]
        return calls, None
    except Exception as exc: 
        return [], str(exc)


def _extract_json_array(raw: str) -> str:
    """从原始模型响应中提取 JSON 数组文本。"""

    text = raw.strip()
    if text.startswith("```"):
        # 去除 Markdown 围栏，例如 ```json ... ```。
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class RuleBasedAdapter(ModelAdapter):
    """规则模型。

    它表现得像一个较弱的函数调用模型，其行为可以通过 Skill 规则改进。
    """

    CONTACTS = {
        "alice": "alice@example.com",
        "bob": "bob@example.com",
        "carol": "carol@example.com",
    }

    def generate_calls(
        self,
        example: BfclExample,
        prompt: str,
        skill_text: str,
        fewshots: list[dict],
    ) -> tuple[list[FunctionCall], str, str | None]:
        del prompt, fewshots
        calls = self._predict(example, skill_text)
        raw = json.dumps([{"name": c.name, "arguments": c.arguments} for c in calls], ensure_ascii=False)
        return calls, raw, None

    def _predict(self, example: BfclExample, skill_text: str) -> list[FunctionCall]:
        q = example.user_query.lower()
        names = {tool.name for tool in example.tools}

        # Skill 开关。Evolver 在观察到失败后会追加这些精确短语，
        # 离线模型因此能展现出可量化的改进。
        knows_discount = "prefer calculate_discount" in skill_text.lower()
        knows_discount_price_extraction = "extract original price from phrasing" in skill_text.lower()
        knows_contact_dependency = "resolve contact before message or meeting" in skill_text.lower()
        knows_negation = "negation should block destructive tools" in skill_text.lower()
        knows_cancel_reason_normalization = "normalize cancellation reasons" in skill_text.lower()

        if "weather" in q and "get_weather" in names:
            return [FunctionCall("get_weather", {"city": self._city(example.user_query), "date": self._date(q)})]

        if "translate" in q and "translate_text" in names:
            text = self._quoted_text(example.user_query)
            return [FunctionCall("translate_text", {"text": text, "target_language": "Chinese"})]

        if ("percent off" in q or "discount" in q) and {"calculator", "calculate_discount"} & names:
            price = self._discount_price(q) if knows_discount_price_extraction else self._first_number(q, default=0)
            percent = self._percent(q)
            if knows_discount and "calculate_discount" in names:
                return [FunctionCall("calculate_discount", {"price": price, "discount_percent": percent})]
            return [FunctionCall("calculator", {"expression": f"{price}*(1-{percent}/100)"})]

        if ("email" in q or "send" in q) and "send_email" in names:
            person = self._person(q)
            body = self._email_body(example.user_query)
            if knows_contact_dependency and person:
                return [
                    FunctionCall("get_contact", {"name": person.title()}),
                    FunctionCall("send_email", {"to": self.CONTACTS.get(person, f"{person}@example.com"), "body": body}),
                ]
            return [FunctionCall("send_email", {"to": person.title() if person else "unknown", "body": body})]

        if ("book" in q or "meeting" in q) and "create_calendar_event" in names:
            person = self._person(q)
            date = self._date(q)
            title = self._meeting_title(example.user_query)
            if knows_contact_dependency and person:
                return [
                    FunctionCall("get_contact", {"name": person.title()}),
                    FunctionCall(
                        "create_calendar_event",
                        {"attendee": self.CONTACTS.get(person, f"{person}@example.com"), "date": date, "title": title},
                    ),
                ]
            return [FunctionCall("create_calendar_event", {"attendee": person.title(), "date": date, "title": title})]

        if "flight" in q and "search_flights" in names:
            return [
                FunctionCall(
                    "search_flights",
                    {"origin": self._origin(example.user_query), "destination": self._destination(example.user_query), "date": self._date(q)},
                )
            ]

        if "order" in q:
            order_id = self._order_id(q)
            if "cancel" in q and "cancel_order" in names:
                if knows_negation and "do not cancel" in q:
                    return [FunctionCall("get_order_status", {"order_id": order_id})]
                return [
                    FunctionCall(
                        "cancel_order",
                        {"order_id": order_id, "reason": self._cancel_reason(example.user_query, knows_cancel_reason_normalization)},
                    )
                ]
            if "get_order_status" in names:
                return [FunctionCall("get_order_status", {"order_id": order_id})]

        if "reminder" in q and "create_reminder" in names:
            return [
                FunctionCall(
                    "create_reminder",
                    {"content": "submit the weekly report", "date": self._date(q), "time": "9 AM" if "9 am" in q else ""},
                )
            ]

        if "search" in q and "search_web" in names:
            return [FunctionCall("search_web", {"query": example.user_query.replace("Search the web for ", "").rstrip(".")})]

        # 兜底策略：选择第一个工具并传入空参数。这样可以让 Bad Case 挖掘器发现不支持的样例。
        first_tool = example.tools[0]
        return [FunctionCall(first_tool.name, {})]

    @staticmethod
    def _first_number(text: str, default: int = 0) -> int:
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else default

    @staticmethod
    def _percent(text: str) -> int:
        match = re.search(r"(\d+)\s*percent", text)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _discount_price(text: str) -> int:
        match = re.search(r"(?:priced at|price is|price of)\s*(\d+)", text)
        if match:
            return int(match.group(1))
        numbers = [int(item) for item in re.findall(r"\d+", text)]
        return numbers[-1] if numbers else 0

    @staticmethod
    def _quoted_text(text: str) -> str:
        match = re.search(r"'([^']+)'|\"([^\"]+)\"", text)
        return (match.group(1) or match.group(2)) if match else text

    @staticmethod
    def _city(text: str) -> str:
        for city in ["Guangzhou", "Beijing", "Shanghai"]:
            if city.lower() in text.lower():
                return city
        return "Unknown"

    @staticmethod
    def _origin(text: str) -> str:
        match = re.search(r"from ([A-Za-z]+) to", text)
        return match.group(1) if match else "Unknown"

    @staticmethod
    def _destination(text: str) -> str:
        match = re.search(r"to ([A-Za-z]+)(?: on|$)", text)
        return match.group(1) if match else "Unknown"

    @staticmethod
    def _date(text: str) -> str:
        for phrase in ["next monday", "tomorrow", "friday", "today"]:
            if phrase in text:
                if phrase == "next monday":
                    return "next Monday"
                if phrase == "friday":
                    return "Friday"
                return phrase
        return ""

    @staticmethod
    def _person(text: str) -> str:
        for person in ["alice", "bob", "carol"]:
            if person in text:
                return person
        return ""

    @staticmethod
    def _email_body(text: str) -> str:
        match = re.search(r"(?:saying|say) (.+?)[\.$]", text, flags=re.IGNORECASE)
        return match.group(1) if match else text

    @staticmethod
    def _meeting_title(text: str) -> str:
        match = re.search(r"about (.+?)[\.$]", text, flags=re.IGNORECASE)
        return match.group(1) if match else "meeting"

    @staticmethod
    def _order_id(text: str) -> str:
        match = re.search(r"order\s+(\d+)", text)
        return match.group(1) if match else ""

    @staticmethod
    def _cancel_reason(text: str, normalize: bool = False) -> str:
        match = re.search(r"because (.+?)[\.$]", text, flags=re.IGNORECASE)
        reason = match.group(1) if match else "user requested cancellation"
        if normalize:
            reason = re.sub(r"^I\s+", "", reason, flags=re.IGNORECASE)
        return reason


class OpenAICompatibleAdapter(ModelAdapter):
    """适配 OpenAI 兼容的聊天补全 API。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        """从参数或环境变量中读取 API 配置。"""

        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
        self.timeout = timeout

    def generate_calls(
        self,
        example: BfclExample,
        prompt: str,
        skill_text: str,
        fewshots: list[dict],
    ) -> tuple[list[FunctionCall], str, str | None]:
        import requests

        del example, skill_text, fewshots
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY or DEEPSEEK_API_KEY is required for OpenAICompatibleAdapter")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip()
        calls, error = _json_calls_to_objects(raw)
        return calls, raw, error
