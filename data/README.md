# Data Directory

This folder holds EEG data for the project.

- `raw/` – Original CHB-MIT `.edf` files downloaded from PhysioNet
- `processed/` – Numpy arrays and labels after windowing, normalization, and train/val/test split

## Steps

1. Create an account on PhysioNet and agree to the CHB-MIT terms.
2. Download the `.edf` files for all subjects.
3. Place them in `data/raw/` as:

```text
data/
└── raw/
    ├── chb01/
    │   ├── chb01_01.edf
    │   ├── ...
    ├── chb02/
    │   ├── chb02_01.edf
    │   └── ...
    └── ...
```

4. Run:

```bash
python src/dataset.py --config configs/config.yaml
```

This will populate `data/processed/` with sliding-window numpy arrays and label files.
