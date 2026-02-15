"""
Prompt 模板加载和渲染模块

负责从配置文件加载 Prompt 模板，并支持模板变量替换。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"


def load_prompts(path: str | Path = DEFAULT_PROMPTS_PATH) -> dict[str, Any]:
    """
    从 YAML 文件加载 Prompt 模板
    
    Args:
        path: Prompt 配置文件路径
        
    Returns:
        包含所有 Prompt 模板的字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        ValueError: 如果配置文件格式不正确
    """
    prompts_path = Path(path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts 配置文件不存在: {prompts_path}")
    
    data = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Prompts 配置必须是字典格式")
    
    return data


def render_prompt(template: str, **kwargs: Any) -> str:
    """
    渲染 Prompt 模板，替换模板变量
    
    Args:
        template: Prompt 模板字符串
        **kwargs: 模板变量键值对
        
    Returns:
        渲染后的 Prompt 字符串
        
    Example:
        >>> template = "分析视频: {video_name}"
        >>> render_prompt(template, video_name="example.mp4")
        '分析视频: example.mp4'
    """
    return template.format(**kwargs)
