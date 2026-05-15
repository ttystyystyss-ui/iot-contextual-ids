# IoT Contextual IDS

This repository stores code, documentation, and datasets for the IoT contextual IDS project.

## Current Model Versions

Two rolling GAT-GRU IDS versions are included:

| Version | Description | Training script | Online script |
|---|---|---|---|
| Single-point response predictor | Predicts one expected future response for each physical variable and detects large prediction errors. | `models/rolling_multistep/train_rolling_multistep_gnn_gru_ids.py` | `models/rolling_multistep/online_rolling_multistep_gnn_gru_ids_mqtt.py` |
| Quantile P10-P90 predictor | Predicts context-dependent normal response intervals using P10/P50/P90 quantile regression. | `models/rolling_quantile/train_rolling_quantile_gnn_gru_ids.py` | `models/rolling_quantile/online_rolling_quantile_gnn_gru_ids_mqtt.py` |

Detailed comparison:

- `docs/model_versions_comparison.md`
- `docs/rolling_multistep_gnn_gru_ids_summary.md`

## Dataset

The aligned dataset used for the current rolling IDS experiments is included at:

- `data/aligned_all_data_clean_delay.csv`

## Training Commands

Single-point version:

```powershell
python models/rolling_multistep/train_rolling_multistep_gnn_gru_ids.py --csv-path data/aligned_all_data_clean_delay.csv --threshold-percentile 99 --epochs 120
```

Quantile version:

```powershell
python models/rolling_quantile/train_rolling_quantile_gnn_gru_ids.py --csv-path data/aligned_all_data_clean_delay.csv --threshold-percentile 99 --epochs 120
```

## Online Detection Commands

Single-point version:

```powershell
python models/rolling_multistep/online_rolling_multistep_gnn_gru_ids_mqtt.py --no-latch
```

Quantile version:

```powershell
python models/rolling_quantile/online_rolling_quantile_gnn_gru_ids_mqtt.py --no-latch
```

## Legacy Packages

Earlier packaged artifacts are still kept in the repository:

- `gnn_gru_ids_package_20260513_201740.zip`
- `collection_scripts_package_20260513_201903.zip`
