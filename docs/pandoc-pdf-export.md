# Pandoc 批量导出 PDF 指南（Video-Transformer）

本文档说明如何把本项目生成的 Markdown 知识笔记批量导出为 PDF，并给出“统一参数配置（一次配置，多文件复用）”的推荐做法。

## 1. 你现在项目里有哪些关键输出？

项目的输出目录由 `config/config.yaml` 决定：

- `system.output_dir`: 默认 `./data/output`

`src/pipeline.py` 会在该目录下固定创建/写入：

- Markdown 笔记：`data/output/documents/{video_id}_knowledge_note.md`
- 蓝图图片：`data/output/blueprints/{video_id}_mind_map.png`
- 质量报告（可选）：`data/output/documents/{video_id}_quality_report.json`

注意：当前生成的 Markdown 在附录里引用蓝图图片时，通常是相对路径 `../blueprints/...`。
因此**导出 PDF 时最好在 `data/output/documents/` 目录下执行 pandoc**（下面会给批处理脚本）。

## 2. 导出 PDF 前：建议先把“生成的 Markdown”切到 PDF Profile

这一步的目标是：让生成出来的 Markdown 更适合 PDF（更少冗余索引、允许块公式/TikZ、并启用质量门禁清理低质段落）。

编辑 `config/config.yaml`：

```yaml
system:
  # 关键：切换到 PDF profile
  note_profile: "pdf"  # default | pdf

  # 关键：启用质量门禁（会在保存 md 前自动清理常见垃圾段落，并写质量报告）
  quality_gates:
    enabled: true
    max_extra_llm_calls: 1

  # 允许块公式（$$...$$）
  pdf_math:
    enable_display_math: true

  # 允许 TikZ（导出 PDF 更稳）
  pdf_diagrams:
    enable_tikz: true

  # PDF 通常不需要“概念索引”大清单
  render:
    include_concept_index: false
```

可选：导出前先跑一次校验（会检查 HTML、模板垃圾、以及 PDF profile 下的 LaTeX 规则）：

```bash
python -m src.tools.validate_note --glob "data/output/documents/*_knowledge_note.md"
```

## 3. 统一参数配置：用 Pandoc Defaults 文件（推荐）

Pandoc 支持 `--defaults <yaml>`：把你平时写在命令行的所有参数集中到一个 YAML 里，后续导出任意文件都复用同一套配置。

本仓库建议使用：

- Defaults 文件：`config/pandoc-pdf.defaults.yaml`
- LaTeX 头文件（启用 TikZ/字体等）：`config/pandoc-preamble.tex`

这两个文件已经在仓库中提供了参考版本，你可以根据机器上已安装的字体微调。

### 3.1 常用 defaults 文件字段解释

- `pdf-engine`: 选择 `xelatex`（推荐）或 `lualatex`
- `toc` / `toc-depth`: 目录
- `number-sections`: 自动编号
- `resource-path`: 图片资源搜索路径
- `include-in-header`: 引入自定义 LaTeX 头文件（TikZ、字体包）
- `variables`: LaTeX 模板变量（字体、页边距、字号、行距等）

## 4. 单文件导出 PDF（使用统一配置）

推荐从项目根目录执行，但用“目录切换”保证图片相对路径不丢：

```bash
cd data/output/documents

pandoc "BV19TKHzUEVs_p6_knowledge_note.md" \
  -o "../pdfs/BV19TKHzUEVs_p6_knowledge_note.pdf" \
  --defaults "../../../config/pandoc-pdf.defaults.yaml"
```

如果你不想 `cd`，也可以在根目录跑，但你需要确保图片路径正确（通常更容易踩坑）。

## 5. 批量导出多个文件（一次配置，多文件复用）

### 5.1 Bash / Linux / macOS / WSL

在项目根目录执行：

```bash
mkdir -p data/output/pdfs

(
  cd data/output/documents || exit 1
  for f in *_knowledge_note.md; do
    base="${f%.md}"
    pandoc "$f" \
      -o "../pdfs/${base}.pdf" \
      --defaults "../../../config/pandoc-pdf.defaults.yaml"
  done
)
```

### 5.2 PowerShell / Windows

在项目根目录执行：

```powershell
New-Item -ItemType Directory -Force data\output\pdfs | Out-Null

Push-Location data\output\documents

Get-ChildItem *_knowledge_note.md | ForEach-Object {
  $in = $_.Name
  $out = "..\pdfs\$($_.BaseName).pdf"
  pandoc $in -o $out --defaults "..\..\..\config\pandoc-pdf.defaults.yaml"
}

Pop-Location
```

## 6. 常见问题与排查

### 6.1 pandoc / xelatex 不存在

先检查：

```bash
pandoc --version
xelatex --version
```

如果缺失，请安装 Pandoc 和 TeX 发行版（TeX Live / MiKTeX）。

### 6.2 中文显示乱码/方块（字体缺失）

这通常是字体没装或字体名不匹配。

解决方法：

1) 安装 Noto CJK 字体（推荐）
2) 修改 `config/pandoc-pdf.defaults.yaml` 的 `variables.mainfont` / `CJKmainfont`

### 6.3 PDF 里图片不显示

由于笔记里常用 `../blueprints/...` 相对路径引用图片，最稳的办法是：

- 在 `data/output/documents/` 目录下运行 pandoc（本文批处理脚本已经这么做）。

### 6.4 TikZ 不生效

确认：

- defaults 文件包含 `include-in-header: config/pandoc-preamble.tex`
- `config/pandoc-preamble.tex` 里有 `\usepackage{tikz}`
- TeX 安装里包含 TikZ（TeX Live 通常没问题；精简安装可能缺包）

## 7. 推荐工作流（生成 → 校验 → 批量导出）

1) 在 `config/config.yaml` 中启用 PDF profile + 质量门禁
2) 运行生成流程，得到 `data/output/documents/*_knowledge_note.md`
3) 运行校验：`python -m src.tools.validate_note --glob "data/output/documents/*_knowledge_note.md"`
4) 运行批量导出脚本（第 5 节）

---

## 附：为什么不用项目 config 直接驱动 pandoc？

目前仓库里还没有内置 `export_pdf` 工具脚本，`system.pdf_typesetting.*` 主要作为“统一参数来源”。
推荐做法是把真正导出用的参数放到 `config/pandoc-pdf.defaults.yaml`，保证“导出侧”的单一真相；
后续如果你希望完全自动化（从 `config/config.yaml` 读取并拼 pandoc 命令），可以再加一个 `src/tools/export_pdf.py`（风格可参考 `src/tools/validate_note.py` 的 `--glob` 批处理模式）。
