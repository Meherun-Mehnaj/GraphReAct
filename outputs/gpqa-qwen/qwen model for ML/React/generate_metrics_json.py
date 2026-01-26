import json
from collections import defaultdict
from datetime import datetime

# File paths
input_file = r"f:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-qwen\qwen model for ML\agot react - Copy\gpqa_agot_react_detailed_traces_rep.jsonl"
output_file = r"f:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-qwen\qwen model for ML\agot react - Copy\RESULT_METRICS.json"

# Initialize counters and collectors
total_problems = 0
correct_count = 0
incorrect_count = 0
domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
answer_distribution = defaultdict(int)
voting_confidence_scores = []
nodes_created_list = []
nodes_evaluated_list = []
nodes_pruned_list = []
actions_performed_list = []
verifications_performed_list = []
verifications_failed_list = []

# Read and analyze the file
print("Analyzing detailed traces file...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            total_problems += 1
            
            # Count correct/incorrect
            is_correct = obj.get('is_correct', False)
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
            
            # Collect domain statistics
            if 'steps' in obj and 'agot_steps' in obj['steps']:
                for step in obj['steps']['agot_steps']:
                    domain = step.get('domain', 'Unknown')
                    domain_stats[domain]['total'] += 1
                    if is_correct:
                        domain_stats[domain]['correct'] += 1
                    break  # Only count first step domain
            
            # Collect answer distribution
            correct_answer = obj.get('correct_answer', 'N/A')
            answer_distribution[correct_answer] += 1
            
            # Collect metrics
            if 'steps' in obj and 'agot_steps' in obj['steps']:
                for step in obj['steps']['agot_steps']:
                    if 'nodes_created' in step:
                        nodes_created_list.append(step['nodes_created'])
                    if 'nodes_evaluated' in step:
                        nodes_evaluated_list.append(step['nodes_evaluated'])
                    if 'nodes_pruned' in step:
                        nodes_pruned_list.append(step['nodes_pruned'])
                    if 'actions_performed' in step:
                        actions_performed_list.append(step['actions_performed'])
                    if 'verifications_performed' in step:
                        verifications_performed_list.append(step['verifications_performed'])
                    if 'verifications_failed' in step:
                        verifications_failed_list.append(step['verifications_failed'])
                    if 'voting_confidence' in step:
                        voting_confidence_scores.append(step['voting_confidence'])
                    break
        except Exception as e:
            print(f"Error processing line: {e}")
            continue

# Calculate statistics
accuracy_percent = (correct_count / total_problems * 100) if total_problems > 0 else 0

def safe_avg(lst):
    return sum(lst) / len(lst) if lst else 0

def safe_min(lst):
    return min(lst) if lst else None

def safe_max(lst):
    return max(lst) if lst else None

avg_nodes_created = safe_avg(nodes_created_list)
avg_nodes_evaluated = safe_avg(nodes_evaluated_list)
avg_nodes_pruned = safe_avg(nodes_pruned_list)
avg_actions = safe_avg(actions_performed_list)
avg_verifications = safe_avg(verifications_performed_list)
avg_verifications_failed = safe_avg(verifications_failed_list)
avg_confidence = safe_avg(voting_confidence_scores)

# Build metrics dictionary
metrics = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "source_file": input_file.split("\\")[-1],
        "report_type": "GPQA-QWEN Model Performance Metrics"
    },
    "overall_performance": {
        "total_problems": total_problems,
        "correct_answers": correct_count,
        "incorrect_answers": incorrect_count,
        "accuracy_percent": round(accuracy_percent, 2)
    },
    "agot_reasoning_metrics": {
        "average": {
            "nodes_created": round(avg_nodes_created, 1),
            "nodes_evaluated": round(avg_nodes_evaluated, 1),
            "nodes_pruned": round(avg_nodes_pruned, 1),
            "actions_performed": round(avg_actions, 1),
            "verifications_performed": round(avg_verifications, 1),
            "verifications_failed": round(avg_verifications_failed, 1),
            "voting_confidence": round(avg_confidence, 4)
        },
        "totals": {
            "total_nodes_created": sum(nodes_created_list),
            "total_nodes_evaluated": sum(nodes_evaluated_list),
            "total_nodes_pruned": sum(nodes_pruned_list),
            "total_actions_performed": sum(actions_performed_list),
            "total_verifications_performed": sum(verifications_performed_list),
            "total_verifications_failed": sum(verifications_failed_list)
        },
        "min_max": {
            "nodes_created": {
                "min": safe_min(nodes_created_list),
                "max": safe_max(nodes_created_list)
            },
            "nodes_evaluated": {
                "min": safe_min(nodes_evaluated_list),
                "max": safe_max(nodes_evaluated_list)
            },
            "voting_confidence": {
                "min": round(safe_min(voting_confidence_scores), 4) if safe_min(voting_confidence_scores) else None,
                "max": round(safe_max(voting_confidence_scores), 4) if safe_max(voting_confidence_scores) else None
            }
        }
    },
    "domain_breakdown": {},
    "correct_answer_distribution": {},
    "performance_by_category": {}
}

# Add domain breakdown
for domain in sorted(domain_stats.keys()):
    stats = domain_stats[domain]
    domain_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    metrics["domain_breakdown"][domain] = {
        "total": stats['total'],
        "correct": stats['correct'],
        "incorrect": stats['total'] - stats['correct'],
        "accuracy_percent": round(domain_acc, 2)
    }

# Add answer distribution
for answer in sorted(answer_distribution.keys()):
    count = answer_distribution[answer]
    percentage = (count / total_problems * 100) if total_problems > 0 else 0
    metrics["correct_answer_distribution"][answer] = {
        "count": count,
        "percentage": round(percentage, 2)
    }

# Categorize performance
if accuracy_percent >= 80:
    performance = "Excellent"
elif accuracy_percent >= 70:
    performance = "Very Good"
elif accuracy_percent >= 60:
    performance = "Good"
elif accuracy_percent >= 50:
    performance = "Satisfactory"
else:
    performance = "Needs Improvement"

metrics["performance_by_category"]["overall_rating"] = performance
metrics["performance_by_category"]["accuracy_range"] = {
    "min": 0,
    "current": round(accuracy_percent, 2),
    "max": 100,
    "target": 63.31
}

# Write JSON report to file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\n✓ JSON metrics report generated successfully!")
print(f"✓ Saved to: {output_file}")
print(f"\n{'SUMMARY':^80}")
print('-' * 80)
print(f"Total Problems:    {total_problems}")
print(f"Correct:           {correct_count}")
print(f"Incorrect:         {incorrect_count}")
print(f"Accuracy:          {accuracy_percent:.2f}%")
print(f"Performance:       {performance}")
print('=' * 80)
