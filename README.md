# Video-Transformer

视频知识蒸馏工具 — 从 Bilibili 视频自动生成结构化精英知识笔记。

下载视频 → 上传至 Gemini → 多模态分析 → 输出 Markdown 知识文档（含术语表、知识蓝图）。

## 架构

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Downloader │     │  号池代理服务      │     │  Google Gemini   │
│  (yt-dlp)   │     │  /sdk/allocate-key│────▶│  Files API       │
│  Bilibili   │     │  /sdk/report-*    │     │  GenerateContent │
└──────┬──────┘     └────────┬─────────┘     └────────┬────────┘
       │                     │ 分配 Key                │ 直连
       ▼                     ▼                         │
┌──────────────────────────────────────────────────────┘
│  ContentAnalyzer (google-genai SDK)
│  1. 从号池获取 API Key
│  2. SDK 直连上传视频 → 获取 file URI
│  3. 发送视频 + Prompt → 获取 JSON 分析结果
│  4. 向号池报告用量
└──────────────┬───────────────────────────────┐
               ▼                               ▼
        AnalysisResult                   Markdown 知识笔记
        (结构化数据)                     (一句话核心/关键结论/
                                         深度解析/术语表/知识蓝图)
```

## 环境要求

- Python 3.8+
- 本地运行的 [gemini-proxy](../gemini-proxy) 号池代理服务（或直接提供 API Key）

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config/config.yaml`：

```yaml
system:
  max_api_calls: 10          # 单次运行最大 API 调用次数
  temp_dir: "./data/temp"
  output_dir: "./data/output"

proxy:
  base_url: "http://localhost:8000"  # 号池代理地址
  timeout: 60

downloader:
  retry_times: 3
  video_format: "mp4"
  max_resolution: 720

analyzer:
  model: "gemini-2.5-flash"
  temperature: 0.7
  max_output_tokens: 8192
  retry_times: 3
  timeout: 120
```

Prompt 模板在 `config/prompts.yaml` 中定义，可自定义分析风格和输出格式。

API 密钥请通过环境变量提供：

- `VT_GEMINI_API_KEY`
- `VT_KIMI_API_KEY`
- `VT_NANO_BANANA_API_KEY`

推荐做法：复制 `env.example` 为 `.env`，写入密钥后直接运行。

```bash
cp env.example .env
```

## 使用方法

### 1. 启动号池代理

```bash
# 在 gemini-proxy 目录下
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. 下载视频

```bash
python demo_downloader.py
```

支持单个 URL 下载或从 `data/input/URL.txt` 批量下载。

### 3. 分析视频

```python
from pathlib import Path
import os
import sys

sys.path.insert(0, "src")

from utils.config import load_config
from utils.counter import APICounter
from utils.logger import setup_logging
from analyzer import ContentAnalyzer

config = load_config()
logger = setup_logging(config["system"]["log_dir"], "analysis.log")
counter = APICounter(max_calls=10)

# 号池模式（自动从代理分配 Key）
analyzer = ContentAnalyzer(config=config, api_counter=counter, logger=logger)

# 或直接指定 Key (建议从环境变量读取)
# analyzer = ContentAnalyzer(
#     config=config,
#     api_counter=counter,
#     logger=logger,
#     api_key=os.environ.get("VT_GEMINI_API_KEY"),
# )

result = analyzer.analyze_video("data/temp/videos/example.mp4")
print(result.to_markdown())
```

### 4. 集成测试前检查

```bash
python check_integration_ready.py
```

验证号池服务连通性、SDK 端点可用性、测试视频文件是否就绪。

## 项目结构

```
Video-Transformer/
├── config/
│   ├── config.yaml          # 系统配置
│   └── prompts.yaml         # Prompt 模板
├── src/
│   ├── analyzer/
│   │   ├── content_analyzer.py  # 核心：视频分析 + Gemini SDK 调用
│   │   ├── models.py            # AnalysisResult / KnowledgeDocument 数据模型
│   │   └── prompt_loader.py     # Prompt 模板加载
│   ├── downloader/
│   │   └── video_downloader.py  # Bilibili 视频下载（yt-dlp）
│   └── utils/
│       ├── config.py            # 配置加载
│       ├── counter.py           # API 调用计数器
│       ├── logger.py            # 日志设置
│       └── proxy.py             # 号池连通性检查
├── data/
│   ├── temp/videos/             # 下载的视频文件
│   └── output/                  # 生成的知识笔记
├── demo_downloader.py           # 下载器演示脚本
├── check_integration_ready.py   # 集成测试准备检查
└── tests/
```

## 号池集成说明

本项目通过号池代理的 SDK 端点获取 API Key，而非通过 HTTP 代理转发请求：

| 步骤 | 调用 | 说明 |
|------|------|------|
| 分配 Key | `POST /sdk/allocate-key` | 获取 `key_id` + 真实 `api_key` |
| 上传视频 | `genai.Client.files.upload()` | SDK 直连 Google API |
| 生成内容 | `genai.Client.models.generate_content()` | SDK 直连 Google API |
| 报告用量 | `POST /sdk/report-usage` | 号池更新 RPD/RPM 计数 |
| 报告错误 | `POST /sdk/report-error` | 号池标记 Key 状态 |

视频数据直传 Google，不经过号池代理，避免内存和带宽瓶颈。

## 测试

```bash
pytest tests/ -v
```
