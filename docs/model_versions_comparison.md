# Rolling GAT-GRU IDS Model Versions

This document summarizes the two current rolling contextual IDS implementations.

## Shared System Goal

Both versions model whether IoT control states and physical responses are contextually consistent.

Control variables:

| Variable | Meaning |
|---|---|
| `light_state` | Philips Hue light state, `0=off`, `1=on` |
| `fan_state` | Dreo fan state, `0=off`, `1=on` |

Physical response variables:

| Variable | Meaning |
|---|---|
| `ambient_light` | Arduino light sensor value |
| `total_power_w` | Tapo P110 total power value, including light and fan power |

Both versions use:

| Component | Setting |
|---|---|
| Input window | Past `10s` |
| Prediction horizon | Future `5s` |
| Target value | Median of future horizon tail, currently steps `3-5` |
| Target type | Response/delta from current value |
| Graph nodes | `light_state`, `fan_state`, `ambient_light`, `total_power_w` |
| Graph edges | `light_state -> ambient_light`, `light_state -> total_power_w`, `fan_state -> total_power_w`, plus self-loops |
| Context feature | `last_off_ambient_median` |
| Future transition handling | Skip samples/windows when global state changes inside the future horizon |

## Version 1: Single-Point Response Predictor

Files:

| File | Purpose |
|---|---|
| `models/rolling_multistep/train_rolling_multistep_gnn_gru_ids.py` | Train the single-point rolling GAT-GRU response predictor |
| `models/rolling_multistep/online_rolling_multistep_gnn_gru_ids_mqtt.py` | Online MQTT IDS using the single-point predictor |

Main idea:

```text
The model predicts one expected response value for each physical target.
```

Outputs:

| Target | Output |
|---|---|
| `ambient_light` | One predicted standardized response |
| `total_power_w` | One predicted standardized response |

Training loss:

```text
Weighted MSE between predicted response and actual response.
```

Online detection:

```text
error = squared difference between predicted response and actual response
```

An alert is raised when:

```text
total_error > total_threshold
OR ambient_error > ambient_threshold
OR power_error > power_threshold
```

Then the high-error condition must persist for the configured `consecutive` count.

Strengths:

| Strength | Explanation |
|---|---|
| Simple and interpretable | One predicted value per physical target |
| Good baseline | Easy to compare against more advanced methods |
| Efficient | Small output head and simple MSE loss |

Limitations:

| Limitation | Explanation |
|---|---|
| Single-point prediction | Normal physical variation around the predicted value may become false positives |
| Global thresholds | Tolerance is learned from validation-set errors, not predicted dynamically per context |
| Switching sensitivity | Small post-switch deviations may exceed target thresholds |

## Version 2: Quantile P10-P90 Response Predictor

Files:

| File | Purpose |
|---|---|
| `models/rolling_quantile/train_rolling_quantile_gnn_gru_ids.py` | Train the P10/P50/P90 rolling GAT-GRU quantile predictor |
| `models/rolling_quantile/online_rolling_quantile_gnn_gru_ids_mqtt.py` | Online MQTT IDS using predicted response intervals |

Main idea:

```text
The model predicts a context-dependent normal response interval instead of a single point.
```

Outputs:

| Target | Outputs |
|---|---|
| `ambient_light` | `P10`, `P50`, `P90` response |
| `total_power_w` | `P10`, `P50`, `P90` response |

The model head is monotonic by construction:

```text
P50 = center
P10 = center - softplus(lower_width)
P90 = center + softplus(upper_width)
```

Therefore:

```text
P10 <= P50 <= P90
```

Training loss:

```text
Pinball quantile loss for P10, P50, and P90.
```

Online detection:

```text
If actual response is inside P10-P90, interval error is zero.
If actual response is outside P10-P90, error is based on the distance outside the interval.
```

Strengths:

| Strength | Explanation |
|---|---|
| Dynamic tolerance | The predicted interval can be wider during naturally variable contexts and narrower during stable contexts |
| Better fit for IDS | "Observed response outside predicted normal range" is intuitive for contextual anomaly detection |
| No manual physical tolerance | The normal range is learned from data rather than hard-coded as watts or sensor units |

Limitations:

| Limitation | Explanation |
|---|---|
| Requires interval calibration | P10-P90 coverage should be checked on validation/normal data |
| Wider intervals can hide weak attacks | If intervals become too wide, subtle anomalies may be missed |
| More outputs | Each physical target has three outputs instead of one |

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

## Dataset

Dataset uploaded with this version:

```text
data/aligned_all_data_clean_delay.csv
```

It contains aligned IoT sensor and device-state records for the rolling contextual IDS experiments.

## Recommended Evaluation

For normal-mode prediction quality:

| Model | Recommended metrics |
|---|---|
| Single-point predictor | MAE, RMSE, validation MSE loss |
| Quantile predictor | P50 MAE/RMSE, P10-P90 interval coverage, average interval width |

For IDS performance:

| Test type | Recommended metrics |
|---|---|
| Normal data | False positive rate |
| Attack data | Detection rate, detection delay |

