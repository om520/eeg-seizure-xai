# Development Guide

This document outlines the implementation roadmap and guides developers in completing the codebase.

## Current Status

✅ **Completed:**
- README.md with comprehensive project overview
- requirements.txt with all dependencies
- configs/config.yaml with hyperparameters
- data/README.md with dataset instructions
- src/model.py with EEGSeizureDetector Transformer architecture

## Remaining Implementation Tasks

### Core Source Files (src/)

The full implementation scaffolds are provided in this document. These need to be added:

#### 1. **src/loss.py**
- FocalLoss implementation for handling class imbalance
- Binary cross-entropy with focal weighting

#### 2. **src/dataset.py**
- EDF file loading using mne library
- Sliding window segmentation (4-second windows, 2-second step)
- Label extraction from seizure annotations
- Train/val/test splitting
- Data normalization and preprocessing

#### 3. **src/train.py**
- Training loop with PyTorch
- AdamW optimizer with CosineAnnealingLR scheduler
- Linear warmup implementation
- AMP (Automatic Mixed Precision) for efficiency
- Gradient accumulation support
- Early stopping with validation monitoring
- Weights & Biases logging

#### 4. **src/evaluate.py**
- Model evaluation on test set
- Threshold tuning for optimal F1 score
- Metrics computation: Accuracy, F1, Sensitivity, Specificity, AUROC, ECE
- CSV export of metrics

#### 5. **src/mc_dropout.py**
- Monte Carlo Dropout inference (20 stochastic passes)
- Epistemic uncertainty estimation
- Mean and standard deviation computation

#### 6. **src/features.py**
- Multi-domain feature extraction per window:
  - Hjorth parameters (mobility, complexity)
  - Higher-order statistics (kurtosis, skewness)
  - Spectral band powers (delta, theta, alpha, beta, gamma)
  - High/low frequency ratio
  - Spectral entropy
  - Approximate entropy (via nolds)
- Aggregation over channels

#### 7. **src/clinical_report.py**
- Per-window clinical report generation
- Seizure probability with 95% confidence interval
- Uncertainty flag assignment
- Feature summary statistics
- CSV output for clinical documentation

### Directory Structure

Create these directories (can be done via adding .gitkeep files or via outputs):

```
outputs/
├── models/          # Checkpoint .pt files
├── plots/           # Training curves, ROC curves, etc.
└── reports/         # metrics.csv, clinical_reports.csv

notebooks/
└── full_pipeline.ipynb   # End-to-end walkthrough notebook
```

## Implementation Priority

1. **High Priority** (Core functionality):
   - src/loss.py
   - src/dataset.py
   - src/features.py

2. **Medium Priority** (Training & inference):
   - src/train.py
   - src/mc_dropout.py
   - src/evaluate.py

3. **Low Priority** (Reporting):
   - src/clinical_report.py
   - notebooks/full_pipeline.ipynb

## Code Quality Standards

When implementing files, follow these guidelines:

- **Docstrings**: All functions must have clear docstrings explaining inputs/outputs
- **Type Hints**: Use type hints for better code documentation
- **Configuration**: All hyperparameters should be read from configs/config.yaml
- **Error Handling**: Include proper error handling and validation
- **Logging**: Use tqdm for progress bars and logging
- **Comments**: Explain non-obvious logic with inline comments

## Testing

Before committing:

1. Verify imports work correctly
2. Test with small dataset samples
3. Check all config paths are correct
4. Run with `--help` to verify argument parsing

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/implement-dataset

# Make changes and commit
git add src/dataset.py
git commit -m "Implement EDF loading and windowing for dataset.py"

# Push to GitHub
git push origin feature/implement-dataset

# Create Pull Request on GitHub
```

## Next Steps

1. Implement src/loss.py and src/features.py (they don't depend on other files)
2. Implement src/dataset.py (depends on config)
3. Implement src/train.py (depends on loss.py, model.py, dataset.py)
4. Implement src/mc_dropout.py and src/evaluate.py
5. Add clinical report generation
6. Create Jupyter notebook with end-to-end pipeline

## Questions?

Refer to the README.md for project overview and architecture diagram.
