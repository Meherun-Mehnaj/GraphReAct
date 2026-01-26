"""
Master Script - Run All Advanced Metrics and Visualizations
Executes comprehensive analysis and generates all reports and charts.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("=" * 80)
    print("INSTALLING REQUIRED PACKAGES")
    print("=" * 80)
    
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("\n✓ All packages installed successfully!\n")
    except Exception as e:
        print(f"\n⚠ Warning: Could not install packages automatically: {e}")
        print("Please install manually: pip install numpy scipy matplotlib seaborn\n")

def run_advanced_metrics():
    """Run advanced metrics analyzer"""
    print("\n" + "=" * 80)
    print("RUNNING ADVANCED METRICS ANALYSIS")
    print("=" * 80 + "\n")
    
    script_path = os.path.join(os.path.dirname(__file__), 'advanced_metrics_analyzer.py')
    
    try:
        subprocess.check_call([sys.executable, script_path])
        print("\n✓ Advanced metrics analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ Error running advanced metrics: {e}")

def run_visualizations():
    """Run visualization generator"""
    print("\n" + "=" * 80)
    print("RUNNING VISUALIZATION GENERATOR")
    print("=" * 80 + "\n")
    
    script_path = os.path.join(os.path.dirname(__file__), 'visualization_generator.py')
    
    try:
        subprocess.check_call([sys.executable, script_path])
        print("\n✓ Visualizations generated successfully!")
    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print(" " * 20 + "COMPREHENSIVE ANALYSIS SUITE")
    print(" " * 15 + "AGoT-ReAct vs ReAct Performance Analysis")
    print("=" * 80 + "\n")
    
    # Step 1: Install requirements
    response = input("Install required packages? (y/n): ").lower()
    if response == 'y':
        install_requirements()
    
    # Step 2: Run advanced metrics
    print("\nStarting analysis pipeline...")
    run_advanced_metrics()
    
    # Step 3: Generate visualizations
    run_visualizations()
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 25 + "ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  📊 Advanced Metrics Reports:")
    print("     - outputs/gpqa-deepseek/Advanced_Metrics_Report_DeepSeek.txt")
    print("     - outputs/gpqa-qwen/Advanced_Metrics_Report_Qwen.txt")
    print("\n  📈 Visualizations:")
    print("     - outputs/gpqa-deepseek/visualizations/")
    print("     - outputs/gpqa-qwen/visualizations/")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
