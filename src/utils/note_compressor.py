from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass
class TopicBlock:
    number: int
    title: str
    explanation: str
    example: str


@dataclass(frozen=True)
class Chapter:
    label: str
    title: str
    topic_nums: list[int]


TOPIC_HEADING_RE = re.compile(r"^####\s+(\d+)\.\s+(.+)$")


def _split_sentences(text: str) -> list[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    sentences: list[str] = []
    buffer: list[str] = []
    for ch in text:
        buffer.append(ch)
        if ch in "。！？.!?":
            sentences.append("".join(buffer).strip())
            buffer = []
    if buffer:
        sentences.append("".join(buffer).strip())
    return [s for s in sentences if s]


def _compact_sentences(text: str, max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    return "".join(sentences[:max_sentences])


def _clean_block_line(line: str) -> str:
    cleaned = line.strip()
    if cleaned.startswith(">"):
        cleaned = cleaned.lstrip(">").strip()
    if cleaned.startswith("-"):
        cleaned = cleaned.lstrip("-").strip()
    return cleaned


def parse_title(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError("Missing title heading")


def parse_summary(lines: list[str]) -> str:
    for idx, line in enumerate(lines):
        if line.strip() == "> 🎯 **一句话核心**":
            for j in range(idx + 1, len(lines)):
                if lines[j].startswith("> "):
                    return lines[j][2:].strip()
            break
    raise ValueError("Missing one-sentence summary")


def parse_mind_map_line(lines: list[str]) -> str:
    for idx, line in enumerate(lines):
        if line.startswith("## 🖼️ 核心图解"):
            for j in range(idx + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate.startswith("!["):
                    return candidate
    raise ValueError("Missing mind map image")


def parse_topics(lines: list[str]) -> list[TopicBlock]:
    topics: list[TopicBlock] = []
    current: TopicBlock | None = None
    explanation_lines: list[str] = []
    example_lines: list[str] = []
    mode: str | None = None

    def flush() -> None:
        nonlocal current, explanation_lines, example_lines, mode
        if not current:
            return
        explanation = " ".join(explanation_lines).strip()
        example = " ".join(example_lines).strip()
        topics.append(
            TopicBlock(
                number=current.number,
                title=current.title,
                explanation=explanation,
                example=example,
            )
        )
        current = None
        explanation_lines = []
        example_lines = []
        mode = None

    for line in lines:
        heading = TOPIC_HEADING_RE.match(line)
        if heading:
            flush()
            current = TopicBlock(
                number=int(heading.group(1)),
                title=heading.group(2).strip(),
                explanation="",
                example="",
            )
            mode = None
            continue

        if not current:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("**💡 原理解析**") or stripped.startswith(
            "**💡 原理拆解**"
        ):
            mode = "explanation"
            continue

        if stripped.startswith("**🌰 举个栗子**") or stripped.startswith(
            "**🌰 自包含示例"
        ):
            mode = "example"
            continue

        if stripped.startswith("**🌰 示例**"):
            mode = "example"
            continue

        if (
            stripped.startswith("**🧩")
            or stripped.startswith("**⚠️")
            or stripped.startswith("**🔗")
        ):
            mode = None
            continue

        if stripped.startswith("**💻"):
            mode = None
            continue

        if (
            stripped.startswith("#### ")
            or stripped.startswith("### ")
            or stripped.startswith("## ")
        ):
            mode = None
            continue

        cleaned = _clean_block_line(stripped)
        if not cleaned:
            continue

        if mode == "explanation":
            explanation_lines.append(cleaned)
        elif mode == "example":
            example_lines.append(cleaned)

    flush()
    return topics


def build_intro(topics: list[TopicBlock]) -> str:
    if not topics:
        return ""
    sentences: list[str] = []
    for topic in topics:
        if topic.explanation:
            sentence = _compact_sentences(topic.explanation, 1)
            if sentence:
                sentences.append(sentence)
        if len(sentences) >= 2:
            break
    if not sentences and topics[0].explanation:
        sentences.append(_compact_sentences(topics[0].explanation, 1))
    return "".join(sentences)


def build_self_check(topic_titles: list[str]) -> list[str]:
    questions: list[str] = []
    templates = [
        "「{topic}」的核心含义是什么？",
        "「{topic}」在图像分类任务中主要解决什么问题？",
        "什么时候更容易遇到「{topic}」相关的困难？",
    ]
    for idx, title in enumerate(topic_titles[:3]):
        questions.append(templates[idx].format(topic=title))
    return questions


def build_output(
    title: str,
    summary: str,
    mind_map_line: str,
    topics: list[TopicBlock],
    max_lines: int,
) -> str:
    chapters: list[Chapter] = [
        Chapter(
            label="第一部分",
            title="图像分类基础与核心挑战",
            topic_nums=list(range(1, 10)),
        ),
        Chapter(
            label="第二部分",
            title="价值、应用与数据驱动范式",
            topic_nums=list(range(10, 16)),
        ),
        Chapter(
            label="第三部分",
            title="常用数据集与小样本学习",
            topic_nums=list(range(16, 24)),
        ),
        Chapter(
            label="第四部分",
            title="最近邻方法与距离度量",
            topic_nums=list(range(24, 38)),
        ),
        Chapter(
            label="第五部分",
            title="超参数选择与评估",
            topic_nums=list(range(38, 46)),
        ),
        Chapter(
            label="第六部分",
            title="高维挑战与改进方向",
            topic_nums=list(range(46, 55)),
        ),
    ]

    topic_map = {topic.number: topic for topic in topics}
    missing = [num for num in range(1, 55) if num not in topic_map]
    if missing:
        raise ValueError(f"Missing topics: {missing}")

    lines: list[str] = [
        f"# {title}",
        "",
        "> 🎯 **一句话核心**",
        f"> {summary}",
        "",
        "## 🖼️ 核心图解",
        mind_map_line,
        "",
    ]

    for chapter in chapters:
        chapter_topics = [topic_map[num] for num in chapter.topic_nums]
        intro = build_intro(chapter_topics)
        lines.append(f"## {chapter.label}：{chapter.title}")
        lines.append("")
        if intro:
            lines.append(intro)
            lines.append("")
        for topic in chapter_topics:
            explanation = _compact_sentences(topic.explanation, 2)
            example = _compact_sentences(topic.example, 1)
            if example:
                lines.append(f"**{topic.title}**：{explanation} 例如：{example}")
            else:
                lines.append(f"**{topic.title}**：{explanation}")
        lines.append("")
        lines.append(f"### 📋 {chapter.label}自测")
        lines.append("")
        for idx, question in enumerate(
            build_self_check([t.title for t in chapter_topics]), start=1
        ):
            lines.append(f"{idx}. {question}")
        lines.append("")

    lines.append("## 📌 覆盖清单 (Coverage Index)")
    lines.append("")
    for chapter in chapters:
        for num in chapter.topic_nums:
            topic_title = topic_map[num].title
            lines.append(f"{num}. {topic_title} — {chapter.label}：{chapter.title}")

    if len(lines) > max_lines:
        raise ValueError(
            f"Compressed note has {len(lines)} lines, exceeds max {max_lines}."
        )

    return "\n".join(lines).rstrip() + "\n"


def run(input_path: Path, output_path: Path, max_lines: int) -> None:
    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = parse_title(lines)
    summary = parse_summary(lines)
    mind_map_line = parse_mind_map_line(lines)
    topics = parse_topics(lines)

    if len(topics) != 54:
        raise ValueError(f"Expected 54 topics, got {len(topics)}")

    output = build_output(title, summary, mind_map_line, topics, max_lines)
    _ = output_path.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress knowledge note")
    _ = parser.add_argument("--input", required=True, type=Path)
    _ = parser.add_argument("--output", required=True, type=Path)
    _ = parser.add_argument("--max-lines", type=int, default=300)
    args = parser.parse_args()

    input_path = cast(Path, args.input)
    output_path = cast(Path, args.output)
    max_lines = cast(int, args.max_lines)
    run(input_path, output_path, max_lines)


if __name__ == "__main__":
    main()
