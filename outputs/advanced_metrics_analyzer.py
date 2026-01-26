"""
Advanced Metrics Analyzer for AGoT-ReAct vs ReAct Comparison
Computes comprehensive performance metrics including statistical significance,
confusion matrices, and detailed error analysis.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from datetime import datetime
import os

class AdvancedMetricsAnalyzer:
    def __init__(self, react_file, agot_react_file, model_name):
        self.react_file = react_file
        self.agot_react_file = agot_react_file
        self.model_name = model_name
        
        self.react_data = []
        self.agot_data = []
        
    def load_data(self):
        """Load data from JSONL files"""
        print(f"Loading data for {self.model_name}...")
        
        with open(self.react_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.react_data.append(json.loads(line))
                except:
                    pass
                    
        with open(self.agot_react_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.agot_data.append(json.loads(line))
                except:
                    pass
        
        print(f"Loaded {len(self.react_data)} ReAct samples and {len(self.agot_data)} AGoT-ReAct samples")
    
    def compute_confusion_matrix(self, data, method_name):
        """Compute confusion matrix and related metrics"""
        # Since we don't have multi-class, we'll compute binary metrics
        correct = sum(1 for d in data if d.get('is_correct', False))
        total = len(data)
        incorrect = total - correct
        
        # For binary: TP=correct, FN=incorrect (assuming all should be correct)
        tp = correct
        fn = incorrect
        # We don't have true negatives/false positives in this context
        
        metrics = {
            'total_samples': total,
            'true_positives': tp,
            'false_negatives': fn,
            'accuracy': (tp / total * 100) if total > 0 else 0,
            'error_rate': (fn / total * 100) if total > 0 else 0
        }
        
        return metrics
    
    def compute_answer_distribution(self, data):
        """Analyze answer distribution"""
        predicted_answers = [d.get('predicted_answer', 'N/A') for d in data]
        correct_answers = [d.get('correct_answer', 'N/A') for d in data]
        
        pred_dist = Counter(predicted_answers)
        correct_dist = Counter(correct_answers)
        
        return {
            'predicted_distribution': dict(pred_dist),
            'correct_distribution': dict(correct_dist)
        }
    
    def compute_step_statistics(self, data):
        """Detailed statistics about reasoning steps"""
        steps_list = []
        for d in data:
            if 'steps' in d and isinstance(d['steps'], list):
                steps_list.append(len(d['steps']))
            elif 'num_steps' in d:
                steps_list.append(d['num_steps'])
        
        if not steps_list:
            return None
            
        return {
            'mean': np.mean(steps_list),
            'median': np.median(steps_list),
            'std': np.std(steps_list),
            'min': min(steps_list),
            'max': max(steps_list),
            'total': sum(steps_list),
            'quartiles': {
                'q25': np.percentile(steps_list, 25),
                'q50': np.percentile(steps_list, 50),
                'q75': np.percentile(steps_list, 75)
            }
        }
    
    def statistical_significance_test(self):
        """Perform statistical significance tests between ReAct and AGoT-ReAct"""
        react_correct = [1 if d.get('is_correct', False) else 0 for d in self.react_data]
        agot_correct = [1 if d.get('is_correct', False) else 0 for d in self.agot_data]
        
        # T-test for accuracy
        t_stat, p_value = stats.ttest_ind(react_correct, agot_correct)
        
        # Chi-square test
        react_success = sum(react_correct)
        react_fail = len(react_correct) - react_success
        agot_success = sum(agot_correct)
        agot_fail = len(agot_correct) - agot_success
        
        contingency_table = [[react_success, react_fail], 
                            [agot_success, agot_fail]]
        chi2, p_chi2 = stats.chi2_contingency(contingency_table)[:2]
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(react_correct) + np.var(agot_correct)) / 2)
        cohens_d = (np.mean(agot_correct) - np.mean(react_correct)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value_ttest': p_value,
            'chi_square': chi2,
            'p_value_chi2': p_chi2,
            'cohens_d': cohens_d,
            'significance_level': 'Significant (p < 0.05)' if p_value < 0.05 else 'Not Significant (p >= 0.05)',
            'effect_size': 'Small' if abs(cohens_d) < 0.5 else ('Medium' if abs(cohens_d) < 0.8 else 'Large')
        }
    
    def error_analysis(self, data, method_name):
        """Analyze errors by question characteristics"""
        errors = [d for d in data if not d.get('is_correct', False)]
        
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(data) * 100 if data else 0,
            'average_steps_on_errors': 0,
            'average_steps_on_correct': 0
        }
        
        # Steps analysis for correct vs incorrect
        error_steps = []
        correct_steps = []
        
        for d in data:
            num_steps = len(d.get('steps', [])) if 'steps' in d else d.get('num_steps', 0)
            if d.get('is_correct', False):
                correct_steps.append(num_steps)
            else:
                error_steps.append(num_steps)
        
        if error_steps:
            error_analysis['average_steps_on_errors'] = np.mean(error_steps)
        if correct_steps:
            error_analysis['average_steps_on_correct'] = np.mean(correct_steps)
        
        return error_analysis
    
    def compute_efficiency_metrics(self):
        """Compute efficiency and cost-related metrics"""
        react_steps = []
        agot_steps = []
        
        for d in self.react_data:
            steps = len(d.get('steps', [])) if 'steps' in d else d.get('num_steps', 0)
            react_steps.append(steps)
            
        for d in self.agot_data:
            steps = len(d.get('steps', [])) if 'steps' in d else d.get('num_steps', 0)
            agot_steps.append(steps)
        
        efficiency = {
            'react_total_steps': sum(react_steps),
            'agot_total_steps': sum(agot_steps),
            'step_reduction': sum(react_steps) - sum(agot_steps),
            'step_reduction_percent': ((sum(react_steps) - sum(agot_steps)) / sum(react_steps) * 100) if sum(react_steps) > 0 else 0,
            'react_avg_steps': np.mean(react_steps) if react_steps else 0,
            'agot_avg_steps': np.mean(agot_steps) if agot_steps else 0,
            'efficiency_gain': ((np.mean(react_steps) - np.mean(agot_steps)) / np.mean(react_steps) * 100) if np.mean(react_steps) > 0 else 0
        }
        
        return efficiency
    
    def generate_comprehensive_report(self):
        """Generate comprehensive metrics report"""
        self.load_data()
        
        report = []
        report.append("=" * 100)
        report.append(f"ADVANCED PERFORMANCE METRICS REPORT - {self.model_name.upper()}")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Confusion Matrix and Basic Metrics
        report.append(f"{'1. CONFUSION MATRIX AND CLASSIFICATION METRICS':^100}")
        report.append("-" * 100)
        
        react_cm = self.compute_confusion_matrix(self.react_data, "ReAct")
        agot_cm = self.compute_confusion_matrix(self.agot_data, "AGoT-ReAct")
        
        report.append(f"\nReAct Performance:")
        report.append(f"  Total Samples:        {react_cm['total_samples']:>6}")
        report.append(f"  Correct Predictions:  {react_cm['true_positives']:>6}")
        report.append(f"  Incorrect Predictions:{react_cm['false_negatives']:>6}")
        report.append(f"  Accuracy:             {react_cm['accuracy']:>6.2f}%")
        report.append(f"  Error Rate:           {react_cm['error_rate']:>6.2f}%")
        
        report.append(f"\nAGoT-ReAct Performance:")
        report.append(f"  Total Samples:        {agot_cm['total_samples']:>6}")
        report.append(f"  Correct Predictions:  {agot_cm['true_positives']:>6}")
        report.append(f"  Incorrect Predictions:{agot_cm['false_negatives']:>6}")
        report.append(f"  Accuracy:             {agot_cm['accuracy']:>6.2f}%")
        report.append(f"  Error Rate:           {agot_cm['error_rate']:>6.2f}%")
        
        # 2. Statistical Significance
        report.append(f"\n{'2. STATISTICAL SIGNIFICANCE TESTING':^100}")
        report.append("-" * 100)
        
        sig_tests = self.statistical_significance_test()
        report.append(f"\nHypothesis Test: AGoT-ReAct vs ReAct")
        report.append(f"  T-statistic:          {sig_tests['t_statistic']:>10.4f}")
        report.append(f"  P-value (t-test):     {sig_tests['p_value_ttest']:>10.6f}")
        report.append(f"  Chi-square statistic: {sig_tests['chi_square']:>10.4f}")
        report.append(f"  P-value (chi-square): {sig_tests['p_value_chi2']:>10.6f}")
        report.append(f"  Result:               {sig_tests['significance_level']}")
        report.append(f"\nEffect Size Analysis:")
        report.append(f"  Cohen's d:            {sig_tests['cohens_d']:>10.4f}")
        report.append(f"  Interpretation:       {sig_tests['effect_size']}")
        
        # 3. Reasoning Step Statistics
        report.append(f"\n{'3. REASONING STEP STATISTICS':^100}")
        report.append("-" * 100)
        
        react_steps = self.compute_step_statistics(self.react_data)
        agot_steps = self.compute_step_statistics(self.agot_data)
        
        if react_steps and agot_steps:
            report.append(f"\n{'Metric':<25} {'ReAct':>15} {'AGoT-ReAct':>15} {'Improvement':>15}")
            report.append("-" * 100)
            report.append(f"{'Mean Steps':<25} {react_steps['mean']:>15.2f} {agot_steps['mean']:>15.2f} {react_steps['mean']-agot_steps['mean']:>15.2f}")
            report.append(f"{'Median Steps':<25} {react_steps['median']:>15.2f} {agot_steps['median']:>15.2f} {react_steps['median']-agot_steps['median']:>15.2f}")
            report.append(f"{'Std Deviation':<25} {react_steps['std']:>15.2f} {agot_steps['std']:>15.2f} {'-':>15}")
            report.append(f"{'Min Steps':<25} {react_steps['min']:>15} {agot_steps['min']:>15} {'-':>15}")
            report.append(f"{'Max Steps':<25} {react_steps['max']:>15} {agot_steps['max']:>15} {'-':>15}")
            report.append(f"{'Total Steps':<25} {react_steps['total']:>15} {agot_steps['total']:>15} {react_steps['total']-agot_steps['total']:>15}")
            
            report.append(f"\nQuartile Analysis:")
            report.append(f"{'Method':<15} {'Q1 (25%)':<15} {'Q2 (50%)':<15} {'Q3 (75%)':<15}")
            report.append("-" * 100)
            report.append(f"{'ReAct':<15} {react_steps['quartiles']['q25']:<15.2f} {react_steps['quartiles']['q50']:<15.2f} {react_steps['quartiles']['q75']:<15.2f}")
            report.append(f"{'AGoT-ReAct':<15} {agot_steps['quartiles']['q25']:<15.2f} {agot_steps['quartiles']['q50']:<15.2f} {agot_steps['quartiles']['q75']:<15.2f}")
        
        # 4. Error Analysis
        report.append(f"\n{'4. ERROR ANALYSIS':^100}")
        report.append("-" * 100)
        
        react_errors = self.error_analysis(self.react_data, "ReAct")
        agot_errors = self.error_analysis(self.agot_data, "AGoT-ReAct")
        
        report.append(f"\n{'Metric':<40} {'ReAct':>15} {'AGoT-ReAct':>15}")
        report.append("-" * 100)
        report.append(f"{'Total Errors':<40} {react_errors['total_errors']:>15} {agot_errors['total_errors']:>15}")
        report.append(f"{'Error Rate (%)':<40} {react_errors['error_rate']:>15.2f} {agot_errors['error_rate']:>15.2f}")
        report.append(f"{'Avg Steps on Correct Answers':<40} {react_errors['average_steps_on_correct']:>15.2f} {agot_errors['average_steps_on_correct']:>15.2f}")
        report.append(f"{'Avg Steps on Incorrect Answers':<40} {react_errors['average_steps_on_errors']:>15.2f} {agot_errors['average_steps_on_errors']:>15.2f}")
        
        # 5. Efficiency Metrics
        report.append(f"\n{'5. EFFICIENCY AND COMPUTATIONAL COST':^100}")
        report.append("-" * 100)
        
        efficiency = self.compute_efficiency_metrics()
        report.append(f"\nComputational Efficiency:")
        report.append(f"  ReAct Total Steps:        {efficiency['react_total_steps']:>10}")
        report.append(f"  AGoT-ReAct Total Steps:   {efficiency['agot_total_steps']:>10}")
        report.append(f"  Step Reduction:           {efficiency['step_reduction']:>10}")
        report.append(f"  Reduction Percentage:     {efficiency['step_reduction_percent']:>10.2f}%")
        report.append(f"\nAverage Efficiency:")
        report.append(f"  ReAct Avg Steps:          {efficiency['react_avg_steps']:>10.2f}")
        report.append(f"  AGoT-ReAct Avg Steps:     {efficiency['agot_avg_steps']:>10.2f}")
        report.append(f"  Efficiency Gain:          {efficiency['efficiency_gain']:>10.2f}%")
        
        # 6. Answer Distribution
        report.append(f"\n{'6. ANSWER DISTRIBUTION ANALYSIS':^100}")
        report.append("-" * 100)
        
        react_dist = self.compute_answer_distribution(self.react_data)
        agot_dist = self.compute_answer_distribution(self.agot_data)
        
        report.append(f"\nCorrect Answer Distribution (Expected):")
        for ans, count in sorted(react_dist['correct_distribution'].items())[:5]:
            report.append(f"  {ans}: {count}")
        
        report.append("\n" + "=" * 100)
        report.append("END OF REPORT")
        report.append("=" * 100)
        
        return '\n'.join(report)
    
    def save_report(self, output_file):
        """Save report to file"""
        report = self.generate_comprehensive_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Advanced metrics report generated successfully!")
        print(f"✓ Saved to: {output_file}")
        print("\nReport Preview:")
        print("=" * 100)
        print(report[:500] + "...\n")


if __name__ == "__main__":
    # DeepSeek Model Analysis
    print("\n" + "="*100)
    print("ANALYZING DEEPSEEK MODEL")
    print("="*100)
    
    deepseek_analyzer = AdvancedMetricsAnalyzer(
        react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-deepseek\React result\gpqa_react_detailed_traces.jsonl",
        agot_react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-deepseek\AGOT-React\gpqa_agot_react_detailed_traces_deepseek.jsonl",
        model_name="DeepSeek"
    )
    
    deepseek_analyzer.save_report(
        r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-deepseek\Advanced_Metrics_Report_DeepSeek.txt"
    )
    
    # Qwen Model Analysis
    print("\n" + "="*100)
    print("ANALYZING QWEN MODEL")
    print("="*100)
    
    qwen_analyzer = AdvancedMetricsAnalyzer(
        react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-qwen\qwen model for ML\React\gpqa_react_detailed_traces_qwen.jsonl",
        agot_react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-qwen\qwen model for ML\agot react\gpqa_agot_react_detailed_traces_qwen.jsonl",
        model_name="Qwen"
    )
    
    qwen_analyzer.save_report(
        r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-qwen\Advanced_Metrics_Report_Qwen.txt"
    )
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
