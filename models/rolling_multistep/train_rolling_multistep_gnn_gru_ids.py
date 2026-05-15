import argparse
import copy
import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CSV_PATH = "aligned_all_data_clean_delay.csv"
ARTIFACT_DIR = Path("artifacts/rolling_multistep_gnn_gru_ids_run")
LAST_OFF_AMBIENT_WINDOW = 30

NODE_NAMES = ["light_state", "fan_state", "ambient_light", "total_power_w"]
CONTROL_NODES = {"light_state", "fan_state"}
PHYSICAL_NODES = ["ambient_light", "total_power_w"]
CONTEXT_COLUMNS = ["last_off_ambient_median"]
AGE_COLUMNS = {
    "light_state": "light_age_seconds", 
    "fan_state": "dreo_age_seconds",
    "ambient_light": "arduino_age_seconds",
    "total_power_w": "tapo_age_seconds",
}
DIRECTED_EDGES = [
    ("light_state", "ambient_light"),
    ("light_state", "total_power_w"),
    ("fan_state", "total_power_w"),
]


@dataclass
class StandardScaler:
    mean: float
    std: float

    def transform(self, value):
        if value is None:
            return None
        return (value - self.mean) / self.std

    def inverse_transform(self, value):
        return value * self.std + self.mean


@dataclass
class RollingConfig:
    csv_path: str
    max_gap_seconds: float
    input_window_seconds: int
    prediction_horizon_seconds: int
    hidden_size: int
    graph_hidden_size: int
    gru_layers: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    train_ratio: float
    threshold_percentile: float
    target_weights: dict


class GraphAttentionLayer(nn.Module):
    def __init__(self, node_feature_size, graph_hidden_size, edge_index, node_count):
        super().__init__()
        self.node_projection = nn.Linear(node_feature_size, graph_hidden_size, bias=False)
        self.attention_projection = nn.Linear(graph_hidden_size * 2, 1, bias=False)
        self.register_buffer("edge_index", edge_index)
        self.node_count = node_count
        self.last_attention = None

    def forward(self, x):
        h = self.node_projection(x)
        source_index = self.edge_index[0]
        target_index = self.edge_index[1]
        source_h = h[:, source_index, :]
        target_h = h[:, target_index, :]
        scores = self.attention_projection(torch.cat([target_h, source_h], dim=-1)).squeeze(-1)
        scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.2)

        output = torch.zeros_like(h)
        attention = torch.zeros_like(scores)
        for target in range(self.node_count):
            mask = target_index == target
            if not torch.any(mask):
                continue
            target_scores = scores[:, mask]
            target_attention = torch.softmax(target_scores, dim=1)
            attention[:, mask] = target_attention
            output[:, target, :] = (source_h[:, mask, :] * target_attention.unsqueeze(-1)).sum(dim=1)
        self.last_attention = attention.detach()
        return output


class RollingMultiStepGATGRU(nn.Module):
    def __init__(
        self,
        node_feature_size,
        context_feature_size,
        graph_hidden_size,
        hidden_size,
        gru_layers,
        dropout,
        target_size,
        edge_index,
    ):
        super().__init__()
        self.gat = GraphAttentionLayer(
            node_feature_size=node_feature_size,
            graph_hidden_size=graph_hidden_size,
            edge_index=edge_index,
            node_count=len(NODE_NAMES),
        )
        self.node_projector = nn.Sequential(
            nn.ReLU(),
            nn.Linear(graph_hidden_size, graph_hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=len(NODE_NAMES) * graph_hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + context_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, target_size),
        )

    def forward(self, x, context=None):
        batch_size, seq_len, node_count, feature_count = x.shape
        x = x.reshape(batch_size * seq_len, node_count, feature_count)
        x = self.gat(x)
        x = self.node_projector(x)
        x = x.reshape(batch_size, seq_len, node_count * x.shape[-1])
        _, hidden = self.gru(x)
        hidden = hidden[-1]
        if context is None:
            context = torch.zeros(batch_size, 0, dtype=hidden.dtype, device=hidden.device)
        return self.head(torch.cat([hidden, context], dim=1))


class RollingDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, context, y, weights = self.samples[index][:4]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def to_float(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def percentile(values, pct):
    values = sorted(
        value for value in values
        if value is not None and not math.isnan(value)
    )
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * pct / 100.0
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def median(values):
    values = sorted(values)
    if not values:
        return None
    return values[len(values) // 2]


def read_rows(csv_path):
    """Read only the columns used by the rolling no-temperature model."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            if not raw.get("time"):
                continue
            light_state = to_float(raw.get("light_state"))
            fan_state = to_float(raw.get("fan_state"))
            ambient_light = to_float(raw.get("ambient_light"))
            total_power_w = to_float(
                raw.get("fan_power_w") or raw.get("total_power_w") or raw.get("power_w")
            )
            required_values = [light_state, fan_state, ambient_light, total_power_w]
            if any(value is None for value in required_values):
                continue

            validity_columns = ["arduino_valid", "light_valid", "tapo_valid", "dreo_valid"]
            if all(column in raw for column in validity_columns):
                if any(str(raw.get(column)) != "1" for column in validity_columns):
                    continue

            rows.append({
                "time": parse_time(raw["time"]),
                "light_state": round(light_state),
                "fan_state": round(fan_state),
                "ambient_light": ambient_light,
                "total_power_w": total_power_w,
                "light_age_seconds": to_float(raw.get("light_age_seconds")) or 0.0,
                "dreo_age_seconds": to_float(raw.get("dreo_age_seconds")) or 0.0,
                "tapo_age_seconds": to_float(raw.get("tapo_age_seconds")) or 0.0,
                "arduino_age_seconds": to_float(raw.get("arduino_age_seconds")) or 0.0,
            })
    rows.sort(key=lambda row: row["time"])
    off_ambient_values = deque(maxlen=LAST_OFF_AMBIENT_WINDOW)
    for row in rows:
        baseline = median(off_ambient_values)
        row["last_off_ambient_median"] = baseline if baseline is not None else row["ambient_light"]
        if row["light_state"] == 0:
            off_ambient_values.append(row["ambient_light"])
    return rows


def split_segments(rows, max_gap_seconds):
    segments = []
    current = []
    previous_time = None
    for row in rows:
        if previous_time is not None:
            gap = (row["time"] - previous_time).total_seconds()
            if gap > max_gap_seconds and current:
                segments.append(current)
                current = []
        current.append(row)
        previous_time = row["time"]
    if current:
        segments.append(current)
    return segments


def build_edge_index():
    edges = []
    for name in NODE_NAMES:
        index = NODE_NAMES.index(name)
        edges.append((index, index))
    for source, target in DIRECTED_EDGES:
        edges.append((NODE_NAMES.index(source), NODE_NAMES.index(target)))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def global_state(row):
    return f"{int(row['light_state'])}{int(row['fan_state'])}"


def transition_count(records):
    count = 0
    for index in range(1, len(records)):
        if global_state(records[index]) != global_state(records[index - 1]):
            count += 1
    return count


def seconds_since_last_transition(records):
    if len(records) < 2:
        return 0
    current_state = global_state(records[-1])
    seconds = 0
    for index in range(len(records) - 2, -1, -1):
        if global_state(records[index]) != current_state:
            break
        seconds += 1
    return seconds


def target_has_state_change(previous_record, target_records):
    previous_state = global_state(previous_record)
    for record in target_records:
        current_state = global_state(record)
        if current_state != previous_state:
            return True
        previous_state = current_state
    return False


def fit_scalers(rows):
    scalers = {}
    for name in NODE_NAMES + CONTEXT_COLUMNS:
        values = [row[name] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        scalers[name] = StandardScaler(mean=mean, std=math.sqrt(variance) or 1.0)
    for age_column in sorted(set(AGE_COLUMNS.values())):
        values = [row[age_column] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        scalers[age_column] = StandardScaler(mean=mean, std=math.sqrt(variance) or 1.0)
    return scalers


def make_node_features(record, previous_record, scalers):
    values = [scalers[name].transform(record[name]) for name in NODE_NAMES]
    if previous_record is None:
        deltas = [0.0 for _ in NODE_NAMES]
    else:
        previous_values = [scalers[name].transform(previous_record[name]) for name in NODE_NAMES]
        deltas = [value - previous_value for value, previous_value in zip(values, previous_values)]

    features = []
    for name, value, delta in zip(NODE_NAMES, values, deltas):
        age_value = scalers[AGE_COLUMNS[name]].transform(record[AGE_COLUMNS[name]])
        node_type = 1.0 if name in CONTROL_NODES else 0.0
        features.append([value, delta, age_value, node_type])
    return features


def build_features(records, scalers):
    features = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index > 0 else None
        features.append(make_node_features(record, previous, scalers))
    return features


def build_context(record, scalers):
    return [scalers["last_off_ambient_median"].transform(record["last_off_ambient_median"])]


def build_targets(records, scalers):
    targets = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index > 0 else record
        targets.append([
            scalers[name].transform(record[name]) - scalers[name].transform(previous[name])
            for name in PHYSICAL_NODES
        ])
    return targets


def build_response_target(previous_record, target_records):
    start = tail_start_index(len(target_records))
    tail_records = target_records[start:]
    target = []
    for name in PHYSICAL_NODES:
        tail_values = sorted(record[name] for record in tail_records)
        median_value = tail_values[len(tail_values) // 2]
        target.append(median_value - previous_record[name])
    return target


def fit_response_scalers(samples):
    scalers = {}
    for index, name in enumerate(PHYSICAL_NODES):
        values = [sample[2][index] for sample in samples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        scalers[name] = StandardScaler(mean=mean, std=math.sqrt(variance) or 1.0)
    return scalers


def transform_response(response, response_scalers):
    return [
        response_scalers[name].transform(response[index])
        for index, name in enumerate(PHYSICAL_NODES)
    ]


def inverse_transform_response(response, response_scalers):
    return [
        response_scalers[name].inverse_transform(response[index])
        for index, name in enumerate(PHYSICAL_NODES)
    ]


def transform_sample_responses(samples, response_scalers):
    transformed = []
    for x, context, y, weights, metadata in samples:
        transformed.append((x, context, transform_response(y, response_scalers), weights, metadata))
    return transformed


def make_samples_for_segment(records, scalers, input_window, horizon, target_weights):
    samples = []
    skipped_future_transition = 0
    if len(records) < input_window + horizon + 1:
        return samples, skipped_future_transition
    features = build_features(records, scalers)
    weights = [target_weights[name] for name in PHYSICAL_NODES]

    for target_start in range(input_window, len(records) - horizon + 1):
        input_records = records[target_start - input_window:target_start]
        previous_target_record = records[target_start - 1]
        target_records = records[target_start:target_start + horizon]
        if target_has_state_change(previous_target_record, target_records):
            skipped_future_transition += 1
            continue

        x = features[target_start - input_window:target_start]
        context = build_context(previous_target_record, scalers)
        y = build_response_target(previous_target_record, target_records)
        metadata = {
            "prediction_time": previous_target_record["time"].isoformat().replace("+00:00", "Z"),
            "target_start_time": target_records[0]["time"].isoformat().replace("+00:00", "Z"),
            "target_end_time": target_records[-1]["time"].isoformat().replace("+00:00", "Z"),
            "current_state": global_state(previous_target_record),
            "input_transition_count": transition_count(input_records),
            "seconds_since_last_transition": seconds_since_last_transition(input_records),
        }
        samples.append((x, context, y, weights, metadata))
    return samples, skipped_future_transition


def make_samples(segments, scalers, input_window, horizon, target_weights):
    samples = []
    summaries = []
    for segment in segments:
        segment_samples, skipped = make_samples_for_segment(
            segment,
            scalers,
            input_window,
            horizon,
            target_weights,
        )
        summaries.append({
            "start": segment[0]["time"].isoformat().replace("+00:00", "Z"),
            "end": segment[-1]["time"].isoformat().replace("+00:00", "Z"),
            "records": len(segment),
            "samples": len(segment_samples),
            "skipped_future_transition": skipped,
        })
        samples.extend(segment_samples)
    return samples, summaries


def weighted_mse(prediction, target, weights):
    error = (prediction - target) ** 2 * weights
    return error.sum() / weights.sum().clamp_min(1e-8)


def tail_start_index(horizon):
    return max(0, horizon // 2)


def evaluate(model, loader, device, scalers):
    model.eval()
    losses = []
    sample_errors = []
    target_errors = {name: [] for name in PHYSICAL_NODES}
    predictions = []
    targets = []
    with torch.no_grad():
        for x, context, y, weights in loader:
            x = x.to(device)
            context = context.to(device)
            y = y.to(device)
            weights = weights.to(device)
            pred = model(x, context).view_as(y)
            loss = weighted_mse(pred, y, weights)
            losses.append(loss.item())
            per_target = (pred - y) ** 2 * weights
            per_sample = per_target.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
            sample_errors.extend(per_sample.detach().cpu().tolist())
            for index, name in enumerate(PHYSICAL_NODES):
                target_errors[name].extend(per_target[:, index].detach().cpu().tolist())
            predictions.extend(pred.detach().cpu().tolist())
            targets.extend(y.detach().cpu().tolist())
    return sum(losses) / len(losses) if losses else float("nan"), sample_errors, target_errors, predictions, targets


def write_prediction_details(path, samples, predictions, targets, response_scalers):
    fields = [
        "prediction_time",
        "target_start_time",
        "target_end_time",
        "current_state",
        "input_transition_count",
        "seconds_since_last_transition",
    ]
    for name in PHYSICAL_NODES:
        fields.extend([f"actual_response_{name}", f"pred_response_{name}", f"err_{name}"])
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample, pred_seq, target_seq in zip(samples, predictions, targets):
            metadata = sample[4]
            row = {**metadata}
            raw_prediction = inverse_transform_response(pred_seq, response_scalers)
            raw_target = inverse_transform_response(target_seq, response_scalers)
            for index, name in enumerate(PHYSICAL_NODES):
                row[f"actual_response_{name}"] = raw_target[index]
                row[f"pred_response_{name}"] = raw_prediction[index]
                row[f"err_{name}"] = (pred_seq[index] - target_seq[index]) ** 2
            writer.writerow(row)


def compute_attention_summary(model):
    attention = model.gat.last_attention
    if attention is None:
        return {}
    mean_attention = attention.mean(dim=0).detach().cpu().tolist()
    edge_index = model.gat.edge_index.detach().cpu()
    summary = {}
    for edge_position, value in enumerate(mean_attention):
        source = NODE_NAMES[int(edge_index[0, edge_position])]
        target = NODE_NAMES[int(edge_index[1, edge_position])]
        if source == target:
            continue
        summary[f"{source} -> {target}"] = value
    return summary


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv_path)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.csv_path}")

    split_index = max(1, int(len(rows) * args.train_ratio))
    train_rows = rows[:split_index]
    val_rows = rows[split_index:]
    scalers = fit_scalers(train_rows)
    train_segments = split_segments(train_rows, args.max_gap_seconds)
    val_segments = split_segments(val_rows, args.max_gap_seconds)
    target_weights = {
        "ambient_light": args.ambient_weight,
        "total_power_w": args.power_weight,
    }
    train_samples_raw, train_summary = make_samples(
        train_segments,
        scalers,
        args.input_window_seconds,
        args.prediction_horizon_seconds,
        target_weights,
    )
    val_samples_raw, val_summary = make_samples(
        val_segments,
        scalers,
        args.input_window_seconds,
        args.prediction_horizon_seconds,
        target_weights,
    )
    if not train_samples_raw or not val_samples_raw:
        raise SystemExit("Not enough samples. Collect more data or reduce window/horizon length.")

    response_scalers = fit_response_scalers(train_samples_raw)
    train_samples = transform_sample_responses(train_samples_raw, response_scalers)
    val_samples = transform_sample_responses(val_samples_raw, response_scalers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RollingMultiStepGATGRU(
        node_feature_size=4,
        context_feature_size=len(CONTEXT_COLUMNS),
        graph_hidden_size=args.graph_hidden_size,
        hidden_size=args.hidden_size,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        target_size=len(PHYSICAL_NODES),
        edge_index=build_edge_index().to(device),
    ).to(device)
    train_loader = DataLoader(RollingDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RollingDataset(val_samples), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, context, y, weights in train_loader:
            x = x.to(device)
            context = context.to(device)
            y = y.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            pred = model(x, context).view_as(y)
            loss = weighted_mse(pred, y, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        train_loss, _train_errors, train_target_errors, _train_predictions, _train_targets = evaluate(model, train_loader, device, scalers)
        val_loss, _val_errors, val_target_errors, _val_predictions, _val_targets = evaluate(model, val_loader, device, scalers)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_target_mean_errors": {
                name: sum(values) / len(values) if values else None
                for name, values in train_target_errors.items()
            },
            "val_target_mean_errors": {
                name: sum(values) / len(values) if values else None
                for name, values in val_target_errors.items()
            },
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_errors, train_target_errors, _train_predictions, _train_targets = evaluate(model, train_loader, device, scalers)
    val_loss, val_errors, val_target_errors, val_predictions, val_targets = evaluate(model, val_loader, device, scalers)
    threshold = percentile(val_errors, args.threshold_percentile)
    target_thresholds = {
        name: percentile(values, args.threshold_percentile)
        for name, values in val_target_errors.items()
    }
    attention_summary = compute_attention_summary(model)
    config = RollingConfig(
        csv_path=args.csv_path,
        max_gap_seconds=args.max_gap_seconds,
        input_window_seconds=args.input_window_seconds,
        prediction_horizon_seconds=args.prediction_horizon_seconds,
        hidden_size=args.hidden_size,
        graph_hidden_size=args.graph_hidden_size,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        threshold_percentile=args.threshold_percentile,
        target_weights=target_weights,
    )
    checkpoint = {
        "model_type": "rolling_multistep_gat_gru_delta_predictor",
        "prediction_target": "tail_median_response",
        "config": asdict(config),
        "node_names": NODE_NAMES,
        "physical_nodes": PHYSICAL_NODES,
        "context_columns": CONTEXT_COLUMNS,
        "node_feature_size": 4,
        "context_feature_size": len(CONTEXT_COLUMNS),
        "last_off_ambient_window": LAST_OFF_AMBIENT_WINDOW,
        "scalers": {name: asdict(scaler) for name, scaler in scalers.items()},
        "response_scalers": {name: asdict(scaler) for name, scaler in response_scalers.items()},
        "target_space": "standardized_response",
        "threshold": threshold,
        "target_thresholds": target_thresholds,
        "error_method": "tail_median_response",
        "response_tail_start_index": tail_start_index(args.prediction_horizon_seconds),
        "state_dict": model.state_dict(),
    }
    model_path = output_dir / "rolling_multistep_gnn_gru_ids.pt"
    torch.save(checkpoint, model_path)
    metrics = {
        "raw_rows": len(rows),
        "train_segments": train_summary,
        "val_segments": val_summary,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "train_error_p95": percentile(train_errors, 95),
        "val_error_p95": percentile(val_errors, 95),
        "threshold": threshold,
        "target_thresholds": target_thresholds,
        "response_scalers": {name: asdict(scaler) for name, scaler in response_scalers.items()},
        "target_space": "standardized_response",
        "error_method": "tail_median_response",
        "response_tail_start_index": tail_start_index(args.prediction_horizon_seconds),
        "last_off_ambient_window": LAST_OFF_AMBIENT_WINDOW,
        "attention_summary": attention_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "attention_summary.json").write_text(json.dumps(attention_summary, indent=2), encoding="utf-8")
    write_prediction_details(output_dir / "val_predictions.csv", val_samples, val_predictions, val_targets, response_scalers)

    print(f"[INFO] Raw rows: {len(rows)}")
    print(f"[INFO] Train samples: {len(train_samples)}")
    print(f"[INFO] Val samples: {len(val_samples)}")
    print(f"[INFO] Input window: {args.input_window_seconds}s")
    print(f"[INFO] Prediction horizon: {args.prediction_horizon_seconds}s")
    print(f"[INFO] Last-off ambient window: {LAST_OFF_AMBIENT_WINDOW}")
    print("[INFO] Response scalers:")
    for name, scaler in response_scalers.items():
        print(f"  {name}: mean={scaler.mean:.6f}, std={scaler.std:.6f}")
    print(f"[INFO] Error method: tail_median_response start_step={tail_start_index(args.prediction_horizon_seconds) + 1}")
    print(f"[INFO] Threshold: {threshold:.6f}")
    print("[INFO] Target thresholds:")
    for name, value in target_thresholds.items():
        print(f"  {name}: {value}")
    print(f"[INFO] Saved model to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train rolling multi-step GAT-GRU IDS without temperature.")
    parser.add_argument("--csv-path", default=CSV_PATH)
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--max-gap-seconds", type=float, default=5.0)
    parser.add_argument("--input-window-seconds", type=int, default=10)
    parser.add_argument("--prediction-horizon-seconds", type=int, default=5)
    parser.add_argument("--graph-hidden-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--threshold-percentile", type=float, default=97.0)
    parser.add_argument("--ambient-weight", type=float, default=1.0)
    parser.add_argument("--power-weight", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
