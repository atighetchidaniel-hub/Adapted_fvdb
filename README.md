# Adapted fVDB NeuralPVS

This repository contains an adapted version of the NeuralPVS training code with the sparse backend migrated from `spconv` to `fvdb`.

## Current Status

- `fvdb` backend implemented
- `spconv` backend removed from the adapted code path
- real-data validation completed on Unity-generated Viking GV/PVV data

Validated model variants:

- `VNet`
- `VNetInterleaved`
- `VNetLighter`
- `VNetLight`
- `OACNNs`
- `OACNNsInterleaved`

## Quick Start

Create the environment:

```bash
bash scripts/setup_conda_fvdb.sh neuralpvs_fvdb
```

Activate it:

```bash
conda activate neuralpvs_fvdb
```

Run the all-model validation sweep:

```bash
bash scripts/run_validation.sh --save-json results/fvdb_validation_summary.json
```

## Main Documents

- Setup guide: `SETUP_CONDA_FVDB.md`
- Migration summary: `FVDB_MIGRATION_SUMMARY.md`
- Progress/status snapshot: `STATUS_UPDATE_2026-04-10.md`

## Dataset Layout

The training code expects datasets in the form:

```text
<root>/
  datasets/
    <dataset_name>/
      gv/
      pvv/
```

For the current real-data validation, the repo also includes a Viking test dataset under:

```text
data_for_test/viking/r30/
  gv/
  pvv/
```

## Validation Scripts

- `scripts/setup_conda_fvdb.sh`: create and populate the conda environment
- `scripts/validate_fvdb_models.py`: run loader, forward, and train-step validation
- `scripts/run_validation.sh`: lightweight wrapper for the validator
- `scripts/setup_and_validate_fvdb.sh`: setup + validation in one run


