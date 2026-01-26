import json
from collections import defaultdict
from datetime import datetime

# File paths
input_file = r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-qwen\qwen model for ML\agot react\gpqa_agot_react_detailed_traces_qwen.jsonl"
output_file = r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\outputs\gpqa-qwen\qwen model for ML\agot react\RESULT_METRICS_gpqa_qwen_agot_react.jsonl"

# Initialize counters and collectors
total_problems = 0
correct_count = 0
incorrect_count = 0
domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
answer_distribution = defaultdict(int)
voting_confidence_scores = []
nodes_created_list = []
nodes_evaluated_list = []
actions_performed_list = []
verifications_performed_list = []

# Read and analyze the file
print("Analyzing detailed traces file...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            total_problems += 1
            
            # Count correct/incorrect
            if obj.get('is_correct'):
                correct_count += 1
            else:
                incorrect_count += 1
            
            # Collect answer distribution
            correct_answer = obj.get('correct_answer', 'N/A')
            answer_distribution[correct_answer] += 1
            
            # Collect ReAct-specific metrics (number of reasoning steps)
            if 'steps' in obj and isinstance(obj['steps'], list):
                num_steps = len(obj['steps'])
                actions_performed_list.append(num_steps)
        except Exception as e:
            print(f"Error processing line: {e}")
            continue

# Calculate statistics
accuracy_percent = (correct_count / total_problems * 100) if total_problems > 0 else 0
avg_steps = sum(actions_performed_list) / len(actions_performed_list) if actions_performed_list else 0

# Build output report
report = []
report.append("=" * 80)
report.append("GPQA-QWEN MODEL PERFORMANCE METRICS REPORT")
report.append("=" * 80)
report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"\n{'OVERALL PERFORMANCE':^80}")
report.append("-" * 80)
report.append(f"Total Problems:           {total_problems:>6}")
report.append(f"Correct Answers:          {correct_count:>6}")
report.append(f"Incorrect Answers:        {incorrect_count:>6}")
report.append(f"Overall Accuracy:         {accuracy_percent:>6.2f}%")
report.append(f"\n{'REASONING METRICS (ReAct)':^80}")
report.append("-" * 80)
report.append(f"Average Steps per Problem:{avg_steps:>6.1f}")

report.append(f"\n{'DOMAIN BREAKDOWN':^80}")
report.append("-" * 80)
report.append("Domain statistics not available for ReAct traces")

report.append(f"\n{'CORRECT ANSWER DISTRIBUTION':^80}")
report.append("-" * 80)
if answer_distribution:
    report.append(f"{'Answer':<10} {'Count':>10} {'Percentage':>15}")
    report.append("-" * 80)
    for answer in sorted(answer_distribution.keys()):
        count = answer_distribution[answer]
        percentage = (count / total_problems * 100) if total_problems > 0 else 0
        report.append(f"{answer:<10} {count:>10} {percentage:>14.2f}%")
else:
    report.append("No answer distribution data available")

report.append(f"\n{'DETAIL STATISTICS':^80}")
report.append("-" * 80)
max_steps = f"{max(actions_performed_list):>6}" if actions_performed_list else "   N/A"
min_steps = f"{min(actions_performed_list):>6}" if actions_performed_list else "   N/A"
total_steps = f"{sum(actions_performed_list):>6}" if actions_performed_list else "   N/A"
report.append(f"Max Steps per Problem:    {max_steps}")
report.append(f"Min Steps per Problem:    {min_steps}")
report.append(f"Total Reasoning Steps:    {total_steps}")
report.append("\n" + "=" * 80)

# Write report to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"\n✓ Metrics report generated successfully!")
print(f"✓ Saved to: {output_file}")
print(f"\n{'SUMMARY':^80}")
print('-' * 80)
print(f"Total Problems:    {total_problems}")
print(f"Correct:           {correct_count}")
print(f"Incorrect:         {incorrect_count}")
print(f"Accuracy:          {accuracy_percent:.2f}%")
print('=' * 80)
