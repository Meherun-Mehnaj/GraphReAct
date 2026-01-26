"""
Visualization Generator for AGoT-ReAct vs ReAct Performance Analysis
Creates comprehensive charts and plots for performance comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class VisualizationGenerator:
    def __init__(self, react_file, agot_react_file, model_name, output_dir):
        self.react_file = react_file
        self.agot_react_file = agot_react_file
        self.model_name = model_name
        self.output_dir = output_dir
        
        self.react_data = []
        self.agot_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        print(f"Loaded {len(self.react_data)} ReAct and {len(self.agot_data)} AGoT-ReAct samples")
    
    def plot_accuracy_comparison(self):
        """Bar chart comparing accuracy"""
        react_acc = sum(1 for d in self.react_data if d.get('is_correct', False)) / len(self.react_data) * 100
        agot_acc = sum(1 for d in self.agot_data if d.get('is_correct', False)) / len(self.agot_data) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['ReAct', 'AGoT-ReAct']
        accuracies = [react_acc, agot_acc]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Accuracy Comparison - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.model_name}_accuracy_comparison.png")
        plt.close()
    
    def plot_step_distribution(self):
        """Histogram of reasoning steps"""
        react_steps = [len(d.get('steps', [])) for d in self.react_data if 'steps' in d]
        agot_steps = [len(d.get('steps', [])) for d in self.agot_data if 'steps' in d]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bins = range(0, max(max(react_steps) if react_steps else 0, max(agot_steps) if agot_steps else 0) + 2)
        
        ax.hist(react_steps, bins=bins, alpha=0.6, label='ReAct', color='#FF6B6B', edgecolor='black')
        ax.hist(agot_steps, bins=bins, alpha=0.6, label='AGoT-ReAct', color='#4ECDC4', edgecolor='black')
        
        ax.set_xlabel('Number of Reasoning Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of Reasoning Steps - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_step_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.model_name}_step_distribution.png")
        plt.close()
    
    def plot_box_plot_steps(self):
        """Box plot for step comparison"""
        react_steps = [len(d.get('steps', [])) for d in self.react_data if 'steps' in d]
        agot_steps = [len(d.get('steps', [])) for d in self.agot_data if 'steps' in d]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = [react_steps, agot_steps]
        bp = ax.boxplot(data_to_plot, labels=['ReAct', 'AGoT-ReAct'], 
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Number of Steps', fontsize=12, fontweight='bold')
        ax.set_title(f'Reasoning Steps Box Plot - {self.model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean markers
        means = [np.mean(react_steps), np.mean(agot_steps)]
        ax.plot([1, 2], means, 'D', color='darkgreen', markersize=8, label='Mean', zorder=3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_steps_boxplot.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.model_name}_steps_boxplot.png")
        plt.close()
    
    def plot_error_analysis(self):
        """Error rate comparison"""
        react_errors = len([d for d in self.react_data if not d.get('is_correct', False)])
        agot_errors = len([d for d in self.agot_data if not d.get('is_correct', False)])
        
        react_correct = len(self.react_data) - react_errors
        agot_correct = len(self.agot_data) - agot_errors
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['ReAct', 'AGoT-ReAct']
        correct = [react_correct, agot_correct]
        incorrect = [react_errors, agot_errors]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x, correct, width, label='Correct', color='#95E1D3', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, incorrect, width, bottom=correct, label='Incorrect', color='#F38181', edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Number of Problems', fontsize=12, fontweight='bold')
        ax.set_title(f'Correct vs Incorrect Answers - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (c, ic) in enumerate(zip(correct, incorrect)):
            total = c + ic
            ax.text(i, c/2, f'{c}\n({c/total*100:.1f}%)', ha='center', va='center', fontweight='bold')
            ax.text(i, c + ic/2, f'{ic}\n({ic/total*100:.1f}%)', ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_error_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.model_name}_error_analysis.png")
        plt.close()
    
    def plot_efficiency_metrics(self):
        """Efficiency comparison chart"""
        react_steps = [len(d.get('steps', [])) for d in self.react_data if 'steps' in d]
        agot_steps = [len(d.get('steps', [])) for d in self.agot_data if 'steps' in d]
        
        metrics = {
            'Total Steps': [sum(react_steps), sum(agot_steps)],
            'Average Steps': [np.mean(react_steps), np.mean(agot_steps)],
            'Max Steps': [max(react_steps), max(agot_steps)]
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            methods = ['ReAct', 'AGoT-ReAct']
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Efficiency Metrics Comparison - {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_efficiency_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.model_name}_efficiency_metrics.png")
        plt.close()
    
    def plot_performance_radar(self):
        """Radar chart for multi-dimensional comparison"""
        from math import pi
        
        react_acc = sum(1 for d in self.react_data if d.get('is_correct', False)) / len(self.react_data) * 100
        agot_acc = sum(1 for d in self.agot_data if d.get('is_correct', False)) / len(self.agot_data) * 100
        
        react_steps = [len(d.get('steps', [])) for d in self.react_data if 'steps' in d]
        agot_steps = [len(d.get('steps', [])) for d in self.agot_data if 'steps' in d]
        
        # Normalize metrics to 0-100 scale
        max_steps = max(max(react_steps), max(agot_steps))
        react_efficiency = 100 - (np.mean(react_steps) / max_steps * 100)
        agot_efficiency = 100 - (np.mean(agot_steps) / max_steps * 100)
        
        categories = ['Accuracy', 'Efficiency', 'Consistency']
        
        react_values = [react_acc, react_efficiency, 100 - np.std(react_steps)/max_steps*100]
        agot_values = [agot_acc, agot_efficiency, 100 - np.std(agot_steps)/max_steps*100]
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        react_values += react_values[:1]
        agot_values += agot_values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, react_values, 'o-', linewidth=2, label='ReAct', color='#FF6B6B')
        ax.fill(angles, react_values, alpha=0.25, color='#FF6B6B')
        
        ax.plot(angles, agot_values, 'o-', linewidth=2, label='AGoT-ReAct', color='#4ECDC4')
        ax.fill(angles, agot_values, alpha=0.25, color='#4ECDC4')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title(f'Performance Radar Chart - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_performance_radar.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.model_name}_performance_radar.png")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization charts"""
        self.load_data()
        
        print(f"\nGenerating visualizations for {self.model_name}...")
        print("=" * 80)
        
        self.plot_accuracy_comparison()
        self.plot_step_distribution()
        self.plot_box_plot_steps()
        self.plot_error_analysis()
        self.plot_efficiency_metrics()
        self.plot_performance_radar()
        
        print("=" * 80)
        print(f"✓ All visualizations generated successfully in: {self.output_dir}\n")


if __name__ == "__main__":
    # DeepSeek Visualizations
    print("\n" + "="*80)
    print("GENERATING DEEPSEEK VISUALIZATIONS")
    print("="*80)
    
    deepseek_viz = VisualizationGenerator(
        react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-deepseek\React result\gpqa_react_detailed_traces.jsonl",
        agot_react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-deepseek\AGOT-React\gpqa_agot_react_detailed_traces_deepseek.jsonl",
        model_name="DeepSeek",
        output_dir=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-deepseek\visualizations"
    )
    
    deepseek_viz.generate_all_visualizations()
    
    # Qwen Visualizations
    print("\n" + "="*80)
    print("GENERATING QWEN VISUALIZATIONS")
    print("="*80)
    
    qwen_viz = VisualizationGenerator(
        react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-qwen\qwen model for ML\React\gpqa_react_detailed_traces_qwen.jsonl",
        agot_react_file=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-qwen\qwen model for ML\agot react\gpqa_agot_react_detailed_traces_qwen.jsonl",
        model_name="Qwen",
        output_dir=r"F:\Data Science\DS\7th Semester\ML\Project\AGoT-ReAct\Math Performance\GraphReAct\outputs\gpqa-qwen\visualizations"
    )
    
    qwen_viz.generate_all_visualizations()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80)
