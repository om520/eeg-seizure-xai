# Explainable & Uncertainty-Aware Transformer for EEG Seizure Detection

This repository implements a **Transformer**-based classifier for EEG-based epileptic seizure detection with Monte Carlo Dropout–based uncertainty estimation, multi-domain feature–based interpretability, and per-window clinical report generation on the CHB-MIT Scalp EEG Database.

## Abstract

We propose an EEG seizure detection pipeline built around a Transformer encoder applied to short EEG windows, trained with Focal Loss to address strong class imbalance between seizure and non-seizure segments. Monte Carlo Dropout at inference yields seizure probabilities and epistemic uncertainty estimates per window, which are combined with multi-domain features (Hjorth parameters, higher-order statistics, spectral band powers, entropy-based descriptors) to support explainability. The system generates per-window clinical reports containing seizure likelihood, confidence intervals, uncertainty flags, and feature summaries, aiming to bridge automated detection with clinician-facing interpretability on the CHB-MIT dataset.

---

## Dataset: CHB-MIT Scalp EEG Database

We use the CHB-MIT Scalp EEG Database hosted on PhysioNet.

- **Official PhysioNet page**: https://physionet.org/content/chbmit/1.0.0/
- The dataset contains long-term scalp EEG recordings from pediatric subjects with intractable seizures, sampled at 256 Hz with mostly 23-channel recordings.

### Download instructions

1. Create a free PhysioNet account and accept the CHB-MIT data use conditions.
2. Download all `.edf` files for each subject (cases `chb01`, `chb02`, …).
3. Place them under:

```text
eeg-seizure-xai/
└── data/
    └── raw/
        ├── chb01/
        │   ├── chb01_01.edf
        │   ├── chb01_02.edf
        │   └── ...
        ├── chb02/
        │   ├── chb02_01.edf
        │   └── ...
        └── ...
```

4. Run: `python src/dataset.py --config configs/config.yaml`

Processed, windowed numpy arrays and labels will be written to `data/processed/`.

---

## Installation

```bash
git clone https://github.com/om520/eeg-seizure-xai.git
cd eeg-seizure-xai
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

All commands assume you run them from the repository root.

### 1. Preprocess EDF files (windowing, labels)

```bash
python src/dataset.py --config configs/config.yaml
```

This will:
- Load `.edf` files from `data/raw/` using `mne`.
- Segment EEG into overlapping 4-second sliding windows.
- Extract window-level labels (seizure vs non-seizure).
- Save numpy arrays and label files under `data/processed/`.

### 2. Train Transformer model

```bash
python src/train.py --config configs/config.yaml
```

This will:
- Instantiate `EEGSeizureDetector` with `d_model`, `nhead`, `num_layers`, `dropout` from the config.
- Train using Focal Loss, AdamW, CosineAnnealingLR with linear warmup, AMP, and gradient accumulation.
- Log training/evaluation metrics and artifacts to Weights & Biases.
- Save checkpoints to `outputs/models/` (e.g., `final_model.pt`).

### 3. Evaluate with Monte Carlo Dropout

```bash
python src/evaluate.py --config configs/config.yaml
```

This will:
- Load the trained model checkpoint.
- Run 20-pass Monte Carlo Dropout per window to estimate seizure probability (`mc_mean`) and uncertainty (`mc_std`).
- Perform threshold tuning and compute metrics (Accuracy, F1, Sensitivity, Specificity, AUROC, ECE).
- Save `metrics.csv` and plots under `outputs/reports/` and `outputs/plots/`.

### 4. Generate per-window clinical report

```bash
python src/clinical_report.py --index 42 --config configs/config.yaml
```

This will:
- Load the specified window (index 42 by default from processed data).
- Run MC Dropout to obtain seizure probability and uncertainty.
- Extract multi-domain features for that window.
- Generate a structured clinical-style report (printed to console and optionally saved as a CSV row in `outputs/reports/`).

---

## Results

Example placeholder table (replace with your actual results after running experiments):

| Metric       | Value |
|-------------|-------|
| Accuracy    | 0.94  |
| F1          | 0.92  |
| Sensitivity | 0.91  |
| Specificity | 0.95  |
| AUROC       | 0.97  |
| ECE         | 0.03  |

---

## Architecture Diagram

```text
          +----------------------------+
          |      EEG Window (4 s)      |
          |   (C channels × T samples) |
          +--------------+-------------+
                         |
                         v
          +----------------------------+
          |     Transformer Encoder    |
          | (stacked encoder layers,   |
          |  multi-head self-attn)     |
          +--------------+-------------+
                         |
                         v
          +----------------------------+
          |       MC Dropout Head      |
          | (20 stochastic forward     |
          |  passes with dropout)      |
          +--------------+-------------+
                         |
                         v
        +----------------+------------------------+
        |                                         |
 n        v                                         v
+---------------------+                 +---------------------+
|  Seizure Probability|                 |    Uncertainty      |
|    (mc_mean)        |                 |     (mc_std)        |
+----------+----------+                 +----------+----------+
           |                                    |
           +----------------+-------------------+
                            |
                            v
                +------------------------+
                |    Feature Extraction  |
                | Hjorth, kurtosis,      |
                | skew, spectral bands,  |
                | spectral entropy, ApEn |
                +------------+-----------+
                             |
                             v
                 +--------------------------+
                 |     Clinical Report      |
                 |  - Seizure prob + CI     |
                 |  - Uncertainty flag      |
                 |  - Feature summary       |
                 +--------------------------+
```

---

## Citation / Acknowledgement

If you use this repository, please cite the CHB-MIT Scalp EEG Database as requested by PhysioNet.

> The CHB-MIT Scalp EEG Database was collected at the Children's Hospital Boston and is distributed via PhysioNet.

Also cite PhysioNet itself following their recommended citation on the dataset page.
