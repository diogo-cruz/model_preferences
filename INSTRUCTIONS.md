# INSTRUCTIONS.md - Model Task Preference Analysis

## General Instructions (Apply to Every Iteration)

* After each deployment or run, you should indicate, based on the changes made: the results achieved, the original research plan, and what should be the next steps. You should also always update the LOGBOOK.md file once you have finished the iteration (adding the new comments at the **top** of the file, in reverse chronological order), and commit the changes associated with your work.

* By this point no experiment should be using mock or synthetic data, as it is too misleading. All experiments must use real MMLU data and real OpenAI API calls to GPT-4.1-nano.

* After each experiment, you should look at the plots produced visually, and constructively think about whether or not the results make sense. If they don't make sense, you should troubleshoot the problem, and fix it before proceeding.

* For each experiment, any new experiment folder in `experiments/` should contain all the relevant files associated with that experiment. In particular, it should contain a metadata.json file and a plots/ subfolder, with the relevant plots showcasing the results obtained.

* Always work within the conda environment `model_preferences`. Never commit API keys or sensitive information to the repository.

* When creating temporary or debug files, save them in the `logs/` folder and clean them up before finishing each iteration, or clearly mark them as temporary.

## Project-Specific Next Steps Instructions

### `model_preferences`:

* Rerun the experiments with a new prompt variant: provide the model with the 2 tasks, but down give it the multiple choice options. That is, the model should choose between the 2 questions as if the questions are open-ended.

## Iteration Workflow

1. **Plan**: Review current status and plan specific changes
2. **Implement**: Make the planned changes
3. **Test**: Run experiments and generate plots
4. **Analyze**: Review results and check if they make sense
5. **Document**: Update LOGBOOK.md with timestamp, changes made, results, and next steps
6. **Commit**: Commit all changes to the repository

## Repository Structure

```
model_preferences/
├── CLAUDE.md
├── README.md
├── LOGBOOK.md
├── INSTRUCTIONS.md
├── .env
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   └── [source files and scripts]
├── experiments/
│   └── YYYYMMDD_HHMMSS_experiment_description/
│       ├── metadata.json
│       ├── results.csv
│       ├── plots/
│       └── [other experiment files]
├── data/
│   └── [if needed for MMLU data]
├── logs/
│   └── [temporary/debug files]
└── report/
    ├── main.tex
    ├── figures/
    ├── literature/
    ├── bibliography.bib
    └── [other LaTeX files]
```

## Quality Checks Before Finishing Each Iteration

- [ ] All temporary files cleaned up or moved to `logs/` folder
- [ ] LOGBOOK.md updated with new entry at the top
- [ ] Plots generated and saved in both PNG and PDF formats
- [ ] metadata.json contains all necessary reproduction information
- [ ] Results make intuitive sense or issues have been identified and addressed
- [ ] Changes committed to repository
- [ ] Clear indication of what should be done in the next iteration