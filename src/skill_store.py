"""Skill 版本管理。

Agent 将可复用经验写入 Skill 文件，
并通过版本号管理，以便在更新带来负面影响时回滚。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class SkillStore:
    """管理单个 SKILL.md 与 few-shot 文件的读取、写入和版本控制。"""

    def __init__(self, skill_dir: str | Path) -> None:
        """初始化 Skill 存储目录及相关文件路径。"""

        self.skill_dir = Path(skill_dir)
        self.skill_path = self.skill_dir / "SKILL.md"
        self.base_skill_path = self.skill_dir / "SKILL_BASE.md"
        self.fewshot_path = self.skill_dir / "fewshots.jsonl"
        self.base_fewshot_path = self.skill_dir / "fewshots_base.jsonl"

        # 版本目录用于保存内容寻址的历史 Skill 副本
        self.versions_dir = self.skill_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def read_skill(self) -> str:
        """读取当前 SKILL.md 的文本内容。"""

        return self.skill_path.read_text(encoding="utf-8")

    def write_skill(self, content: str) -> str:
        """写入 SKILL.md 并为其创建一个新的内容寻址版本。"""

        self.skill_path.write_text(content, encoding="utf-8")
        return self.snapshot(content)

    def snapshot(self, content: str | None = None) -> str:
        """保存一份内容寻址的 Skill 副本，并返回其版本 ID。"""

        if content is None:
            content = self.read_skill()
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
        version_path = self.versions_dir / f"SKILL_{digest}.md"
        if not version_path.exists():
            version_path.write_text(content, encoding="utf-8")
        return digest

    def append_fewshot(self, item: dict) -> None:
        """将一个已验证的示例追加到 few-shot 记忆中。

        该方法会执行简单的去重检查。避免 prompt 越来越长，却不会带来新信息。
        """

        existing = self.read_fewshots(limit=10_000)
        fingerprint = json.dumps({"user_query": item.get("user_query"), "calls": item.get("calls")}, sort_keys=True)
        for old_item in existing:
            old_fingerprint = json.dumps(
                {"user_query": old_item.get("user_query"), "calls": old_item.get("calls")},
                sort_keys=True,
            )
            if old_fingerprint == fingerprint:
                return

        with self.fewshot_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def read_fewshots(self, limit: int = 5) -> list[dict]:
        """读取最近的 limit 条 few-shot 示例。"""

        if not self.fewshot_path.exists():
            return []
        items: list[dict] = []
        with self.fewshot_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items[-limit:]

    def reset(self) -> None:
        """重置可变的 Skill 记忆，使实验可以重复运行。"""

        if self.base_skill_path.exists():
            self.skill_path.write_text(self.base_skill_path.read_text(encoding="utf-8"), encoding="utf-8")
        if self.base_fewshot_path.exists():
            self.fewshot_path.write_text(self.base_fewshot_path.read_text(encoding="utf-8"), encoding="utf-8")
