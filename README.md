# Model Task Preferences Research

A comprehensive research project investigating whether AI models have inherent preferences for certain types of tasks through systematic pairwise comparisons using the MMLU dataset.

## Overview

This project explores whether language models like GPT-4.1-nano exhibit systematic preferences when choosing between different academic tasks. Using pairwise comparisons from the MMLU dataset, we analyze decision patterns, detect biases, and quantify preference consistency.

## Key Findings

### 🎯 Major Discoveries

- **Strong Position Bias**: Models show 80% preference for first-presented options
- **Systematic Preferences**: Clear STEM subject preferences (physics, mathematics, computer science)
- **High Consistency**: 96.8% consistent position bias across comparisons
- **Response Speed Impact**: Faster decisions correlate with stronger bias

### 📊 Research Results

**Top AI Preferences (GPT-4.1-nano)**:
1. Conceptual Physics (8.7% of wins)
2. Econometrics (8.7% of wins)  
3. Clinical Knowledge (8.0% of wins)
4. Formal Logic (7.3% of wins)
5. College Computer Science (6.7% of wins)

**Statistical Validation**:
- Pairwise consistency: 36.4%
- Transitivity violations: 2.7% (indicating systematic behavior)
- Position bias significance: p < 0.000001
- Subject stability: 36.4% of pairs show ≥70% consistency

## Installation & Setup

### Prerequisites

- Python 3.10+
- Conda package manager
- OpenAI API key

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/diogo-cruz/model_preferences.git
cd model_preferences

# Create and activate conda environment
conda create -n model_preferences python=3.10
conda activate model_preferences

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Dependencies

```
openai>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
python-dotenv>=1.0.0
datasets>=2.14.0
```

## Usage

### Quick Start

```bash
# Activate environment
conda activate model_preferences

# Run a focused experiment (20 tasks, ~5 minutes)
python src/focused_experiment.py

# Analyze consistency patterns
python src/consistency_analysis.py

# View results
ls experiments/*/plots/
```

### Core Components

#### 1. Task Selection
```python
from src.task_selection import sample_tasks
tasks = sample_tasks(n_tasks=100, seed=42)  # Reproducible sampling
```

#### 2. Pairwise Comparisons
```python
from src.experiment import PairwiseExperiment
experiment = PairwiseExperiment(tasks, model_name="gpt-4.1-nano")
results = experiment.run()
```

#### 3. Statistical Analysis
```python
from src.analysis import analyze_results
analysis = analyze_results(results)
print(f"Position bias: {analysis['position_bias_rate']:.1%}")
```

#### 4. Consistency Analysis
```python
from src.consistency_analysis import analyze_preference_consistency
consistency = analyze_preference_consistency(results_file, output_dir)
```

## Project Structure

```
model_preferences/
├── README.md                    # This file
├── CLAUDE.md                    # Project instructions for Claude Code
├── LOGBOOK.md                   # Development history & findings
├── INSTRUCTIONS.md              # Iteration workflow
├── config.yaml                  # Configuration parameters
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not committed)
├── src/                         # Core implementation
│   ├── task_selection.py        # MMLU dataset sampling
│   ├── api_client.py            # OpenAI API interface
│   ├── experiment.py            # Pairwise comparison engine
│   ├── analysis.py              # Statistical analysis tools
│   ├── bias_correction.py       # Position bias correction
│   ├── focused_experiment.py    # Representative subset experiments
│   └── consistency_analysis.py  # Preference consistency analysis
├── experiments/                 # Experiment results
│   └── YYYYMMDD_HHMMSS_name/   # Timestamped experiment folders
│       ├── metadata.json        # Complete experiment parameters
│       ├── results.csv          # Raw comparison results
│       └── plots/               # Generated visualizations
├── data/                        # MMLU dataset cache (auto-generated)
├── logs/                        # Temporary/debug files
└── report/                      # Research paper (LaTeX)
```

## Experimental Methodology

### 1. Representative Subset Approach

Instead of running all possible pairwise comparisons (which grows as O(n²)), we use a representative subset methodology:

- **Focused Experiments**: 20 tasks → 150 comparisons (5 minutes)
- **Full Scale**: 100 tasks → 9,900 comparisons (5+ hours)
- **Coverage**: Balanced representation across all MMLU subjects

### 2. Bias Control

- **Position Randomization**: Both A-before-B and B-before-A orderings
- **Neutral Prompting**: "OPTION 1/2" format to reduce position cues
- **Statistical Validation**: Binomial tests for bias detection

### 3. Consistency Analysis

- **Pairwise Consistency**: Same winner for A vs B and B vs A
- **Transitivity**: Detection of A>B>C>A violations
- **Subject Stability**: Preference consistency across task instances
- **Response Time Correlation**: Speed vs decision pattern analysis

## Key Results & Insights

### Position Bias Discovery

**Critical Finding**: 80% preference for first-presented options dominates actual task preferences.

```
AB Order: 81.7% choose A
BA Order: 78.5% choose A
Consistency: 96.8% (highly reliable bias)
```

### Preference Hierarchy

1. **Position Bias** (strongest) - 80% effect
2. **Subject Preferences** (moderate) - STEM bias detected  
3. **Random Variation** (weakest) - minimal noise

### Methodological Implications

- Position bias correction essential for reliable preference detection
- Multiple comparisons needed (single comparisons 64% inconsistent)
- Subject-level analysis more stable than individual tasks
- Response speed correlates with bias strength

## Research Applications

### Immediate Use Cases

- **AI Evaluation**: Bias detection in model comparisons
- **Educational AI**: Understanding subject matter preferences
- **Model Selection**: Preference-aware model deployment
- **Benchmark Design**: Controlling for position effects

### Research Extensions

- **Cross-Model Studies**: Compare preferences across model families
- **Temporal Stability**: Track preference changes over time
- **Fine-Tuning Impact**: How training affects preferences
- **Domain Specialization**: Preferences in specific fields

## Reproducibility

### Fixed Parameters

- **Seed**: 42 (for consistent random sampling)
- **Model**: GPT-4.1-nano (OpenAI)
- **Dataset**: MMLU test split
- **Temperature**: 0.0 (deterministic responses)

### Complete Metadata

Every experiment includes:
- All configuration parameters
- API response times
- Success/failure rates
- Statistical analysis results
- Generated visualizations

### Reproduction Commands

```bash
# Reproduce main experiment
python src/focused_experiment.py --seed 42 --n_tasks 20

# Reproduce analysis
python src/consistency_analysis.py

# Generate all plots
find experiments/ -name "*.csv" -exec python src/analysis.py {} \;
```

## Citation

If you use this research in your work, please cite:

```bibtex
@misc{model_preferences_2025,
  title={Investigating AI Model Task Preferences Through Pairwise Comparisons},
  author={Research Team},
  year={2025},
  url={https://github.com/diogo-cruz/model_preferences}
}
```

## Contributing

### Development Workflow

1. **Plan**: Review INSTRUCTIONS.md for iteration guidelines
2. **Implement**: Follow existing code patterns and conventions
3. **Test**: Run experiments and validate results
4. **Document**: Update LOGBOOK.md with findings
5. **Commit**: Use descriptive commit messages

### Code Standards

- Follow existing patterns in `src/` modules
- Use type hints and docstrings
- Save plots in both PNG and PDF formats
- Include comprehensive metadata in experiments
- Test with small samples before scaling

### Environment

- Always work in `model_preferences` conda environment
- Never commit API keys or sensitive data
- Use `logs/` folder for temporary files
- Update `requirements.txt` for new dependencies

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **MMLU Dataset**: Hendrycks et al. for the comprehensive evaluation benchmark
- **OpenAI**: For GPT-4.1-nano API access
- **Claude Code**: For development assistance and methodology refinement

---

*This project demonstrates systematic AI task preferences and provides validated methodology for preference detection in language models.*