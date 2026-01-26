# GraphReAct - Math Performance Analysis

This repository contains the implementation and evaluation of **AGoT-ReAct** (Algorithm of Thoughts + ReAct) framework for mathematical reasoning tasks, compared against baseline ReAct methods.

## 📁 Project Structure

```
.
├── AGoT_main/                  # Core AGoT implementation
│   ├── agot.py                # Main AGoT algorithm
│   └── run_gpqa.py            # GPQA dataset runner
│
├── AGoT-ReAct/                # AGoT-ReAct notebooks
│   ├── Math-AGoT-ReAct_kaggle_v2.ipynb
│   ├── Math-AGoT-ReAct_kaggle_v4.ipynb
│   ├── math-agot-react-kaggle-v3.ipynb
│   └── requirements.txt
│
├── ReAct/                     # Baseline ReAct implementation
│   ├── Math-ReAct-colab.ipynb
│   ├── Math-ReAct-kaggle.ipynb
│   └── requirements.txt
│
├── ReAct-master/              # Original ReAct framework
│   ├── alfworld.ipynb
│   ├── FEVER.ipynb
│   ├── hotpotqa.ipynb
│   ├── WebShop.ipynb
│   ├── wikienv.py
│   ├── wrappers.py
│   └── prompts/
│
└── outputs/                   # Experimental results
    ├── ALL/                   # Combined comparisons
    ├── gpqa-deepseek/        # DeepSeek model results
    │   ├── AGOT-React/
    │   └── React result/
    └── gpqa-qwen/            # Qwen model results
        └── qwen model for ML/
            ├── agot react/
            └── React/
```

## 🎯 Overview

This project implements and evaluates the **AGoT-ReAct** approach, which combines:
- **Algorithm of Thoughts (AGoT)**: Advanced reasoning framework for complex problem-solving
- **ReAct**: Reasoning and Acting paradigm for LLM agents

The framework is evaluated on mathematical reasoning tasks using the GPQA dataset with multiple LLM backends.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r AGoT-ReAct/requirements.txt
```

### Running Experiments

#### AGoT-ReAct
```bash
# Using Kaggle notebooks
jupyter notebook AGoT-ReAct/Math-AGoT-ReAct_kaggle_v4.ipynb
```

#### Baseline ReAct
```bash
# Using Kaggle notebooks
jupyter notebook ReAct/Math-ReAct-kaggle.ipynb
```

## 📊 Results

Results are organized by model and approach:

- **DeepSeek Model**
  - AGoT-ReAct: `outputs/gpqa-deepseek/AGOT-React/`
  - ReAct: `outputs/gpqa-deepseek/React result/`

- **Qwen Model**
  - AGoT-ReAct: `outputs/gpqa-qwen/qwen model for ML/agot react/`
  - ReAct: `outputs/gpqa-qwen/qwen model for ML/React/`

- **Combined Analysis**: `outputs/ALL/Combined_Results_Comparison_FULL.txt`

Each results directory contains:
- Detailed traces (`.jsonl`)
- Performance metrics
- Analysis scripts (`generate_metrics.py`)

## 📈 Evaluation Metrics

The evaluation scripts compute:
- Accuracy
- Reasoning trace quality
- Step-by-step performance analysis

## 🔧 Tools & Technologies

- **Language Models**: DeepSeek, Qwen
- **Dataset**: GPQA (General Purpose Question Answering)
- **Framework**: ReAct, Algorithm of Thoughts
- **Environment**: Jupyter Notebooks, Python

## 📝 License

See [LICENSE](ReAct-master/LICENSE) for details.

## 🤝 Contributing

This is a research project for ML coursework. For questions or collaboration, please open an issue.

## 📚 References

This project builds upon:
- ReAct: Synergizing Reasoning and Acting in Language Models
- Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models

---

**Course**: Machine Learning (7th Semester)  
**Institution**: Data Science Program
