import json
from pathlib import Path
from statistics import mean
from datetime import datetime
from collections import defaultdict

INPUT = Path(r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-deepseek\AGOT-React\gpqa_agot_react_detailed_traces_deepseek.jsonl")
OUTPUT = Path(r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-deepseek\AGOT-React\Results_gpqa_agot_react_deepseek.jsonl")


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


def extract_domain(record):
    """Extract domain from question or other fields"""
    question = record.get("question", "").lower()
    
    # Look for domain keywords in the question
    domain_keywords = {
        "physics": ["exoplanet", "star", "stellar", "orbit", "planet", "velocity", "temperature", 
                   "spectral", "photosphere", "effective temperature", "solar", "mass", "radius",
                   "gravity", "motion", "force", "acceleration", "kinetic", "potential", "momentum",
                   "energy", "pressure", "quantum", "relativity", "optics", "wave", "frequency",
                   "symmetry", "lorentz", "poincare", "smeft", "coupling", "gauge"],
        "astronomy": ["galaxy", "cosmic", "redshift", "light-year", "parsec", "observation", "telescope",
                     "celestial", "nebula", "pulsar", "quasar", "black hole"],
        "chemistry": ["molecule", "atom", "chemical", "bond", "reaction", "element", "compound",
                     "ph", "titration", "acid", "base", "salt", "ion", "edta", "nmr", "ftir",
                     "spectroscopy", "oxidation", "reduction", "electrode", "electrolysis", 
                     "equilibrium", "concentration", "molar", "molality", "solution", "solvent",
                     "polymer", "organic", "inorganic", "synthesis", "hydrocarbon", "functional group",
                     "grubbs", "catalyst", "rearrangement", "cope", "cyclohexane", "reagent",
                     "ene", "aldehyde", "methyl", "vinyl", "alkyne", "alkene"],
        "biology": ["gene", "protein", "cell", "organism", "species", "enzyme", "dna", "rna",
                   "mutation", "evolution", "organism", "organism", "photosynthesis", "metabolism",
                   "mitochondria", "chloroplast", "antibody", "antigen"],
        "computer science": ["algorithm", "output", "input", "code", "program", "variable", "logic",
                            "data structure", "array", "database", "network", "coding"],
        "mathematics": ["equation", "theorem", "proof", "calculation", "formula", "matrix",
                       "derivative", "integral", "function", "polynomial", "trigonometric"],
    }
    
    # Count keyword matches
    matches = {}
    for domain, keywords in domain_keywords.items():
        count = sum(1 for kw in keywords if kw in question)
        if count > 0:
            matches[domain] = count
    
    # Return domain with highest matches, or "Unknown"
    if matches:
        return max(matches, key=matches.get)
    
    return "Unknown"


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

        # Domain extraction using improved method
        domain = extract_domain(r)

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
        lines.append(f"{'Domain':<30} {'Total':>8} {'Correct':>10} {'Accuracy':>12} {'% of Total':>12}")
        lines.append("-" * 80)
        for dom in sorted(domain_stats.keys()):
            d = domain_stats[dom]
            tot = d["total"]
            cor = d["correct"]
            acc = (cor / tot * 100) if tot else 0.0
            pct_of_total = (tot / total * 100) if total else 0.0
            lines.append(f"{dom:<30} {tot:>8} {cor:>10} {acc:>11.2f}% {pct_of_total:>11.2f}%")
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
