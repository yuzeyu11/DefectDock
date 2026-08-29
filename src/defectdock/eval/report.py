"""Customer acceptance report generation (Markdown).

Produces the acceptance-facing report used in industrial delivery: detection
rate / false discovery rate / per-class breakdown / threshold recommendation /
miss & false-positive statistics. The Markdown output is fed to
``tools/md2docx.py`` for the final ``.docx`` deliverable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .cv import EvalSummary, ThresholdScan
from .miss_analysis import MissReport, group_by_class


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_acceptance_report(
    summary: EvalSummary,
    *,
    scan: ThresholdScan | None = None,
    miss_report: MissReport | None = None,
    title: str = "AI 视觉检测模型验收报告",
    target_recall: float = 0.95,
    serious_class: str = "冲孔",
) -> str:
    """Render a customer acceptance report as Markdown."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## 1. 总体指标（验收口径）")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|------|----|------|")
    lines.append(f"| 检出率（Recall） | {_pct(summary.recall)} | 真实缺陷中被正确检出的比例，目标 ≥{_pct(target_recall)} |")
    lines.append(f"| 假发现比例（FDR，1-Precision） | {_pct(summary.false_discovery_rate)} | 检出结果中误报的比例，越低越好 |")
    lines.append(f"| 精确率（Precision） | {_pct(summary.precision)} | 检出结果中正确命中的比例 |")
    lines.append(f"| 真实缺陷框 | {summary.gt_total} | 测试集标注框总数 |")
    lines.append(f"| 正确检出（TP） | {summary.tp} | — |")
    lines.append(f"| 漏检（FN） | {summary.fn} | 漏检代价最高，需重点关注 |")
    lines.append(f"| 误检（FP） | {summary.fp} | 误检=过杀，影响产线效率 |")
    lines.append("")

    lines.append("## 2. 逐类明细")
    lines.append("")
    lines.append("| 类别 | 真实框 | 正确检出 | 漏检 | 误检 | 检出率 | FDR |")
    lines.append("|------|-------:|-------:|-----:|-----:|-------:|-------:|")
    for item in summary.per_class:
        lines.append(
            f"| {item.name} | {item.gt} | {item.tp} | {item.fn} | {item.fp} | "
            f"{_pct(item.recall)} | {_pct(item.false_discovery_rate)} |"
        )
    lines.append("")

    if scan is not None and scan.recommended is not None:
        rec = scan.recommended
        lines.append("## 3. 阈值工作点推荐")
        lines.append("")
        lines.append(f"- **推荐置信度阈值**：{rec.threshold}")
        lines.append(f"- 该阈值下：检出率 {_pct(rec.recall)}，假发现比例 {_pct(rec.false_discovery_rate)}")
        lines.append(f"- 目标检出率：{_pct(scan.target_recall)}（达到/未达到）")
        lines.append("")

    if miss_report is not None:
        lines.append("## 4. 误检 / 漏检样本统计")
        lines.append("")
        lines.append(f"- 漏检：{miss_report.summary.get('missed', 0)} 处")
        lines.append(f"- 误检：{miss_report.summary.get('false_positive', 0)} 处")
        lines.append(f"- 低置信度命中：{miss_report.summary.get('low_confidence', 0)} 处")
        missed_by_class = group_by_class(miss_report.missed)
        fp_by_class = group_by_class(miss_report.false_positives)
        if missed_by_class:
            lines.append("")
            lines.append("漏检类别分布：")
            for name, count in sorted(missed_by_class.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {name}：{count} 处")
        if fp_by_class:
            lines.append("")
            lines.append("误检类别分布：")
            for name, count in sorted(fp_by_class.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {name}：{count} 处")
        lines.append("")

    lines.append("## 5. 验收结论")
    lines.append("")
    passed = summary.recall >= target_recall
    lines.append(f"- 检出率 **{_pct(summary.recall)}** {'≥' if passed else '<'} 目标 {_pct(target_recall)}：**{'达标' if passed else '未达标'}**")
    serious = next((item for item in summary.per_class if item.name == serious_class), None)
    if serious is not None:
        serious_pass = serious.recall >= 0.99
        lines.append(f"- 严重缺陷「{serious_class}」检出率 **{_pct(serious.recall)}**：**{'达标（≥99%）' if serious_pass else '未达标'}**")
    lines.append("")
    lines.append("> 说明：本报告指标来自指定测试集，生产验收须使用现场独立采集数据复验。")
    lines.append("")
    return "\n".join(lines)


def write_report(markdown: str, path: str | Path) -> Path:
    """Write the report Markdown to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return out
