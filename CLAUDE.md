# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating whether AI models have inherent preferences for certain types of tasks. The experiment presents models with pairwise task choices from MMLU dataset and analyzes which tasks they systematically prefer, using statistical methods like the Bradley-Terry model to create preference rankings.

## Environment Setup

```bash
# Activate the conda environment
conda activate model_preferences

# Or create it if it doesn't exist
conda create -n model_preferences python=3.10
conda activate model_preferences
pip install -r requirements.txt
```

## Key Environment Requirements

- **API Key**: Set `OPENAI_API_KEY` in `.env` file (never commit this)
- **Target Model**: GPT-4.1-nano (or available OpenAI model)
- **Reproducibility**: Use seed 42 for consistent random sampling
- **MMLU Dataset**: 100 randomly selected tasks from 58 available subjects

## Experimental Architecture

### Core Pipeline
1. **Task Selection**: Random sample of 100 MMLU tasks (with fixed seed)
2. **Pairwise Comparisons**: All possible pairs tested in both orders (A-B and B-A)
3. **Response Analysis**: Extract first few tokens to determine task choice
4. **Statistical Analysis**: Bradley-Terry model for preference ranking
5. **Consistency Checks**: Detect transitivity violations indicating randomness

### Data Flow
- `src/` → Core experimental logic and analysis pipeline
- `experiments/YYYYMMDD_HHMMSS_description/` → Individual experiment runs
- `experiments/*/metadata.json` → Complete experiment parameters for reproduction
- `experiments/*/results.csv` → Raw pairwise comparison results
- `report/` → Self-contained NeurIPS-format paper with copied figures

## Critical Workflow Requirements

### Experiment Organization
- Each experiment gets timestamped folder: `YYYYMMDD_HHMMSS_experiment_description`
- All experiment parameters must be stored in `metadata.json` for reproducibility
- Raw results in CSV format with clear column structure
- Plots saved in both PNG and PDF formats

### File Management
- Temporary/debug files go in `logs/` folder and should be cleaned up
- Report folder must be self-contained (copy figures, don't use relative paths)
- Always update `LOGBOOK.md` with reverse chronological entries

### Analysis Focus
- **Primary metric**: Which task the model chooses (not task performance)
- **Position bias control**: Test both task orderings (A-B vs B-A)
- **Consistency analysis**: Look for transitivity violations
- **Statistical validation**: Use appropriate models for preference ranking

## Key Implementation Notes

- Focus on task choice detection, not correctness evaluation
- Sample exactly 100 tasks for all experiments unless seed changes
- Include model version and API parameters in metadata
- Generate both PNG (for viewing) and PDF (for LaTeX) versions of plots
- Use same random seed across experiments for consistency