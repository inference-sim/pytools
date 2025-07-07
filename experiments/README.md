# Experiments

This folder contains all experimental work for the vLLM simulator project. Each experiment is documented to ensure reproducibility and knowledge sharing.

## Folder Structure

```
data/                            # Shared experimental scripts
experiments/
├── README.md                    # This file
├── experiment_docs/             # Individual experiment documentation
│   ├── experiment_001_data_collection.md
│   ├── experiment_002_regression_model.md
│   └── ...
├── step_time_analysis.ipynb
├── predict_model_execute.ipynb
├── sim_data_analysis.ipynb
├── utils.py                     # Common helper functions
└── ...
 
```

## Purpose

This experiments folder serves to:
- Document the research process and findings
- Ensure all experiments can be reproduced
- Share knowledge and build upon previous work
- Validate simulator performance against real vLLM

## Experiment Documentation Template

Each experiment document in `experiment_docs/` follows this structure:

### File Naming
`exp_XX_brief_description.md`

### Document Structure
```markdown
# Experiment XXX: [Title]

**Date**:
**Author**: [Name]
**Status**: [Planned/In Progress/Completed]

## Purpose/Goal
What this experiment aims to achieve.
Link to GitHub issues if relevant.

## How to Reproduce
**Code Changes**: Files modified and key code snippets
**Configuration**: Config files used, commit references
**Data Collection**: What data was collected and where it's stored
**Commands**: Step-by-step commands to reproduce

## Analysis
**Analysis Performed**: What analysis was done
**Code/Files**: Reference to notebooks or analysis files used

## Key Takeaways
**Findings**: Main discoveries and quantitative results
**Implications**: What these results mean for the project
**Future Work**: Suggested follow-up experiments
```

## Getting Started

**For New Experiments**:
1. Create experiment document in `experiment_docs/`
2. Implement changes and collect data
3. Analyze results in dedicated notebooks
4. Update documentation with findings

**For Reproducing Experiments**:
1. Read the experiment documentation
2. Follow the reproduction steps exactly
3. Compare results with documented findings
