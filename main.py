#!/usr/bin/env python3
"""
Video-Transformer 主程序

视频内容知识化与图谱生成系统
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import BatchResult
from pipeline import VideoPipeline
from utils.config import load_config
from utils.counter import APICounter
from utils.logger import setup_logging
from utils.progress_tracker import ProgressTracker
from utils.proxy import verify_proxy_connection

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("提示: 安装 'rich' 库以获得更好的输出体验: pip install rich")


class VideoTransformerCLI:
    """CLI 应用主类"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None

    def print(self, *args, **kwargs):
        """美化打印"""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def run(self, args: argparse.Namespace) -> int:
        """
        运行主程序

        Args:
            args: 命令行参数

        Returns:
            退出码 (0=成功, 1=失败)
        """
        try:
            # 1. 加载配置
            self.print("\n[bold blue]🔧 加载配置文件...[/bold blue]")
            config = load_config(args.config)

            # 覆盖配置(如果命令行指定)
            if args.output_dir:
                config["system"]["output_dir"] = args.output_dir
            if args.max_api_calls:
                config["system"]["max_api_calls"] = args.max_api_calls

            # 2. 初始化日志
            log_dir = config["system"]["log_dir"]
            logger = setup_logging(log_dir, "main.log")
            logger.info("=" * 60)
            logger.info("Video-Transformer 启动")
            logger.info("=" * 60)

            # 3. 健康检查
            self.print("[bold blue]🏥 系统健康检查...[/bold blue]")
            if not self._health_check(config, logger):
                return 1

            # 4. 初始化组件
            api_counter = APICounter(max_calls=config["system"]["max_api_calls"])

            # 进度追踪器(如果需要)
            progress_tracker = None
            if not args.no_checkpoint:
                progress_file = Path(config["system"]["temp_dir"]) / "progress.json"
                progress_tracker = ProgressTracker(progress_file, logger)

            # 流程编排器
            pipeline = VideoPipeline(
                config=config,
                logger=logger,
                api_counter=api_counter,
                progress_tracker=progress_tracker,
            )

            # 5. 处理视频
            if args.url:
                # 单个视频
                result = pipeline.process_single_video(args.url)
                self._print_single_result(result)
                return 0 if result.success else 1

            elif args.batch:
                # 批量处理
                urls = self._load_url_list(args.batch)
                if not urls:
                    self.print("[bold red]❌ URL 列表为空[/bold red]")
                    return 1

                self.print(f"\n[bold green]📋 加载了 {len(urls)} 个视频 URL[/bold green]")

                # 过滤已处理(如果启用断点续传)
                if progress_tracker:
                    # 提取视频 ID
                    video_ids = [pipeline._extract_video_id(url) for url in urls]
                    unprocessed_ids = progress_tracker.filter_unprocessed(video_ids)
                    urls = [
                        url
                        for url, vid in zip(urls, video_ids)
                        if vid in unprocessed_ids
                    ]

                    if not urls:
                        self.print(
                            "[bold yellow]✅ 所有视频已处理完成![/bold yellow]"
                        )
                        return 0

                    self.print(f"[yellow]剩余待处理: {len(urls)} 个[/yellow]")

                batch_result = pipeline.process_batch(urls)
                self._print_batch_result(batch_result)
                return 0 if batch_result.failed == 0 else 1

            else:
                self.print("[bold red]❌ 请指定 --url 或 --batch 参数[/bold red]")
                return 1

        except KeyboardInterrupt:
            self.print("\n[bold yellow]⚠️  用户中断[/bold yellow]")
            return 1
        except Exception as e:
            self.print(f"[bold red]❌ 程序异常: {e}[/bold red]")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

    def _health_check(self, config: dict, logger) -> bool:
        """系统健康检查"""
        # 检查代理号池连通性
        proxy_url = config.get("proxy", {}).get("base_url", "http://localhost:8000")

        self.print(f"  检查代理号池服务 ({proxy_url})...")

        if verify_proxy_connection(proxy_url):
            self.print("  [green]✅ 代理号池连接正常[/green]")
        else:
            self.print(
                f"  [yellow]⚠️  代理号池服务不可用 ({proxy_url})[/yellow]"
            )
            self.print(
                "  [yellow]提示: 如果配置文件中有固定 API Key,程序仍可运行[/yellow]"
            )

        # 检查输出目录
        output_dir = Path(config["system"]["output_dir"])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            self.print(f"  [green]✅ 创建输出目录: {output_dir}[/green]")
        else:
            self.print(f"  [green]✅ 输出目录存在: {output_dir}[/green]")

        return True

    def _load_url_list(self, file_path: str) -> list[str]:
        """加载 URL 列表"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
            return urls
        except Exception as e:
            self.print(f"[bold red]❌ 加载 URL 文件失败: {e}[/bold red]")
            return []

    def _print_single_result(self, result):
        """打印单个视频处理结果"""
        if RICH_AVAILABLE:
            table = Table(title="处理结果")
            table.add_column("项目", style="cyan")
            table.add_column("值", style="green")

            table.add_row("视频 ID", result.video_id)
            table.add_row(
                "状态", "✅ 成功" if result.success else f"❌ 失败: {result.error_message}"
            )
            if result.success:
                table.add_row("文档路径", result.document_path or "N/A")
                table.add_row("蓝图路径", result.blueprint_path or "N/A")
                table.add_row("审核分数", f"{result.audit_score:.1f}")
            table.add_row("API 调用", str(result.api_calls_used))
            table.add_row("耗时", f"{result.processing_time:.1f}s")

            self.console.print(table)
        else:
            print(f"\n{'='*60}")
            print(result)
            print(f"{'='*60}\n")

    def _print_batch_result(self, result: BatchResult):
        """打印批量处理结果"""
        if RICH_AVAILABLE:
            # 摘要表
            summary_table = Table(title="批量处理摘要")
            summary_table.add_column("指标", style="cyan")
            summary_table.add_column("值", style="green")

            summary_table.add_row("总视频数", str(result.total))
            summary_table.add_row("成功", f"[green]{result.successful}[/green]")
            summary_table.add_row("失败", f"[red]{result.failed}[/red]")
            summary_table.add_row("成功率", f"{result.successful/result.total*100:.1f}%")
            summary_table.add_row("总 API 调用", str(result.total_api_calls))
            summary_table.add_row("总耗时", f"{result.total_time:.1f}s")

            self.console.print(summary_table)

            # 详细结果
            if result.results:
                detail_table = Table(title="详细结果")
                detail_table.add_column("视频 ID", style="cyan")
                detail_table.add_column("状态", style="white")
                detail_table.add_column("API 调用", style="yellow")
                detail_table.add_column("耗时", style="magenta")

                for r in result.results:
                    status = (
                        "[green]✅ 成功[/green]"
                        if r.success
                        else f"[red]❌ {r.error_message[:20]}...[/red]"
                    )
                    detail_table.add_row(
                        r.video_id,
                        status,
                        str(r.api_calls_used),
                        f"{r.processing_time:.1f}s",
                    )

                self.console.print(detail_table)
        else:
            print(f"\n{'='*60}")
            print(result)
            print(f"{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="视频内容知识化与图谱生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个视频
  python main.py --url "https://www.bilibili.com/video/BV1xx411c7mD"

  # 批量处理
  python main.py --batch data/input/URL.txt

  # 指定配置文件
  python main.py --config config/custom.yaml --batch data/input/URL.txt

  # 禁用断点续传
  python main.py --batch data/input/URL.txt --no-checkpoint
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        help="单个视频 URL",
    )

    parser.add_argument(
        "--batch",
        type=str,
        metavar="PATH",
        help="批量处理文件路径(每行一个 URL)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        metavar="PATH",
        help="配置文件路径(默认: config/config.yaml)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="PATH",
        help="输出目录(覆盖配置文件)",
    )

    parser.add_argument(
        "--max-api-calls",
        type=int,
        metavar="N",
        help="API 调用上限(默认: 10)",
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="禁用断点续传功能",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细日志模式",
    )

    args = parser.parse_args()

    # 验证参数
    if not args.url and not args.batch:
        parser.print_help()
        sys.exit(1)

    if args.url and args.batch:
        print("错误: --url 和 --batch 不能同时使用")
        sys.exit(1)

    # 运行 CLI
    cli = VideoTransformerCLI()
    exit_code = cli.run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
