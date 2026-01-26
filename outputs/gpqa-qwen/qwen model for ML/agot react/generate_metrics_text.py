import json
from pathlib import Path
from statistics import mean
from datetime import datetime
from collections import defaultdict

INPUT = Path(r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-qwen\qwen model for ML\agot react\gpqa_agot_react_detailed_traces_qwen.jsonl")
OUTPUT = Path(r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-qwen\qwen model for ML\agot react\Results_gpqa_agot_react_qwen.jsonl")


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # skip malformed rows
                print(f"Skipping malformed line: {exc}")
                continue


def main():
    records = list(load_records(INPUT))
    total = len(records)
    correct = sum(1 for r in records if r.get("is_correct"))
    incorrect = total - correct

    react_steps_counts = []
    answer_counts = {}
    total_steps = 0
    domain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in records:
        steps = r.get("steps") or {}
        react_steps = steps.get("react_steps") or []
        agot_steps = steps.get("agot_steps") or []
        react_steps_counts.append(len(react_steps))
        total_steps += len(react_steps)
        ans = r.get("correct_answer", "N/A")
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

        # Domain extraction (prefer structured steps)
        domain = None
        if isinstance(agot_steps, list) and agot_steps:
            d = agot_steps[0].get("domain") if isinstance(agot_steps[0], dict) else None
            if isinstance(d, str) and d.strip():
                domain = d.strip()
        if not domain:
            # Fallback: try react_trace header line like "Domain: Physics"
            rt = r.get("react_trace")
            if isinstance(rt, str):
                for line in rt.splitlines():
                    if line.startswith("Domain:"):
                        domain = line.split(":", 1)[1].strip()
                        break
        if not domain:
            domain = "Unknown"

        domain_stats[domain]["total"] += 1
        if r.get("is_correct"):
            domain_stats[domain]["correct"] += 1

    avg_steps = mean(react_steps_counts) if react_steps_counts else 0
    min_steps = min(react_steps_counts) if react_steps_counts else 0
    max_steps = max(react_steps_counts) if react_steps_counts else 0

    lines = []
    lines.append("=" * 80)
    lines.append("GPQA-QWEN MODEL PERFORMANCE METRICS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"{'OVERALL PERFORMANCE':^80}")
    lines.append("-" * 80)
    lines.append(f"Total Problems:              {total:>6}")
    lines.append(f"Correct Answers:             {correct:>6}")
    lines.append(f"Incorrect Answers:           {incorrect:>6}")
    accuracy = (correct / total * 100) if total else 0
    lines.append(f"Overall Accuracy:         {accuracy:>6.2f}%")
    lines.append("")
    lines.append(f"{'REASONING METRICS (ReAct)':^80}")
    lines.append("-" * 80)
    lines.append(f"Average Steps per Problem:   {avg_steps:4.1f}")
    lines.append("")
    lines.append(f"{'DOMAIN BREAKDOWN':^80}")
    lines.append("-" * 80)
    if domain_stats:
        lines.append(f"{'Domain':<30} {'Total':>8} {'Correct':>10} {'Accuracy':>12}")
        lines.append("-" * 80)
        for dom in sorted(domain_stats.keys()):
            d = domain_stats[dom]
            tot = d["total"]
            cor = d["correct"]
            acc = (cor / tot * 100) if tot else 0.0
            lines.append(f"{dom:<30} {tot:>8} {cor:>10} {acc:>11.2f}%")
    else:
        lines.append("No domain data available")
    lines.append("")
    lines.append(f"{'CORRECT ANSWER DISTRIBUTION':^80}")
    lines.append("-" * 80)
    lines.append(f"{'Answer':<10} {'Count':>10} {'Percentage':>15}")
    lines.append("-" * 80)
    for ans in sorted(answer_counts.keys()):
        count = answer_counts[ans]
        pct = (count / total * 100) if total else 0
        lines.append(f"{ans:<10} {count:>10} {pct:>14.2f}%")
    lines.append("")
    lines.append(f"{'DETAIL STATISTICS':^80}")
    lines.append("-" * 80)
    lines.append(f"Max Steps per Problem: {max_steps:9}")
    lines.append(f"Min Steps per Problem: {min_steps:9}")
    lines.append(f"Total Reasoning Steps: {total_steps:9}")
    lines.append("")
    lines.append("=" * 80)

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
