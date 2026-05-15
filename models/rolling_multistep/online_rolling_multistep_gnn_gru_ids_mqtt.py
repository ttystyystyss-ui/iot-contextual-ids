import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
import torch

from train_rolling_multistep_gnn_gru_ids import (
    NODE_NAMES,
    PHYSICAL_NODES,
    RollingMultiStepGATGRU,
    StandardScaler,
    build_response_target,
    build_edge_index,
    build_context,
    build_features,
    global_state,
    inverse_transform_response,
    seconds_since_last_transition,
    target_has_state_change,
    transform_response,
    transition_count,
)


MODEL_PATH = "artifacts/rolling_multistep_gnn_gru_ids_run/rolling_multistep_gnn_gru_ids.pt"
MQTT_BROKER = "172.24.224.223"
MQTT_PORT = 1883
MQTT_TOPICS = [
    "iot/arduino",
    "iot/light_state",
    "iot/tapo_p110",
    "iot/dreo_fan",
]
ALERT_TOPIC = "iot/rolling_multistep_gnn_gru_ids_alert"
ONLINE_LOG = Path("artifacts/rolling_multistep_gnn_gru_ids_online_log.csv")
POLL_INTERVAL = 1.0
LAST_OFF_AMBIENT_WINDOW = 30

ARDUINO_TTL = 5.0
LIGHT_TTL = 5.0
TAPO_TTL = 5.0
DREO_TTL = 5.0

LOG_FIELDS = [
    "time",
    "status",
    "prediction_id",
    "prediction_time",
    "target_start_time",
    "target_end_time",
    "skipped_reason",
    "history",
    "required_history",
    "anomaly",
    "high_error",
    "total_high_error",
    "ambient_high_error",
    "power_high_error",
    "consecutive_high_error",
    "error",
    "threshold",
    "current_state",
    "light_state",
    "fan_state",
    "input_transition_count",
    "seconds_since_last_transition",
    "err_ambient_light",
    "err_total_power_w",
    "th_ambient_light",
    "th_total_power_w",
    "actual_ambient_light",
    "actual_total_power_w",
    "pred_ambient_light",
    "pred_total_power_w",
    "actual_delta_ambient_light",
    "actual_delta_total_power_w",
    "pred_delta_ambient_light",
    "pred_delta_total_power_w",
    "last_off_ambient_median",
    "light_age_seconds",
    "dreo_age_seconds",
    "tapo_age_seconds",
    "arduino_age_seconds",
]

latest_state = {
    "arduino": {"recv_ts": None, "ambient_light": None},
    "light": {"recv_ts": None, "light_state": None},
    "tapo": {"recv_ts": None, "total_power_w": None},
    "dreo": {"recv_ts": None, "fan_state": None},
}


def response_tail_start(horizon):
    return max(0, horizon // 2)


def iso_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_fresh(recv_ts, ttl):
    return recv_ts is not None and (time.time() - recv_ts) <= ttl


def age_seconds(recv_ts):
    if recv_ts is None:
        return None
    return max(0.0, time.time() - recv_ts)


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_state(value):
    value = to_float(value)
    if value is None:
        return None
    return 1 if value >= 0.5 else 0


def build_current_record():
    arduino_valid = is_fresh(latest_state["arduino"]["recv_ts"], ARDUINO_TTL)
    light_valid = is_fresh(latest_state["light"]["recv_ts"], LIGHT_TTL)
    tapo_valid = is_fresh(latest_state["tapo"]["recv_ts"], TAPO_TTL)
    dreo_valid = is_fresh(latest_state["dreo"]["recv_ts"], DREO_TTL)
    if not (arduino_valid and light_valid and tapo_valid and dreo_valid):
        return None

    record = {
        "time": iso_utc_now(),
        "light_state": to_state(latest_state["light"]["light_state"]),
        "fan_state": to_state(latest_state["dreo"]["fan_state"]),
        "ambient_light": to_float(latest_state["arduino"]["ambient_light"]),
        "total_power_w": to_float(latest_state["tapo"]["total_power_w"]),
        "light_age_seconds": age_seconds(latest_state["light"]["recv_ts"]),
        "dreo_age_seconds": age_seconds(latest_state["dreo"]["recv_ts"]),
        "tapo_age_seconds": age_seconds(latest_state["tapo"]["recv_ts"]),
        "arduino_age_seconds": age_seconds(latest_state["arduino"]["recv_ts"]),
    }
    required_columns = [
        "light_state",
        "fan_state",
        "ambient_light",
        "total_power_w",
        "light_age_seconds",
        "dreo_age_seconds",
        "tapo_age_seconds",
        "arduino_age_seconds",
    ]
    if any(record[column] is None for column in required_columns):
        return None
    return record


class OnlineRollingMultiStepIDS:
    def __init__(self, checkpoint, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = checkpoint["config"]
        self.input_window = int(config["input_window_seconds"])
        self.horizon = int(config["prediction_horizon_seconds"])
        self.threshold = args.threshold if args.threshold is not None else float(checkpoint["threshold"])
        self.target_thresholds = checkpoint.get("target_thresholds", {})
        self.target_weights = config["target_weights"]
        self.scalers = {
            name: StandardScaler(**values)
            for name, values in checkpoint["scalers"].items()
        }
        self.response_scalers = {
            name: StandardScaler(**values)
            for name, values in checkpoint.get("response_scalers", {}).items()
        }
        for name in PHYSICAL_NODES:
            self.response_scalers.setdefault(name, StandardScaler(mean=0.0, std=1.0))
        node_feature_size = checkpoint.get(
            "node_feature_size",
            4,
        )
        self.model = RollingMultiStepGATGRU(
            node_feature_size=node_feature_size,
            context_feature_size=checkpoint.get("context_feature_size", 1),
            graph_hidden_size=config["graph_hidden_size"],
            hidden_size=config["hidden_size"],
            gru_layers=config["gru_layers"],
            dropout=config["dropout"],
            target_size=len(PHYSICAL_NODES),
            edge_index=build_edge_index().to(self.device),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.records = []
        self.pending_predictions = []
        self.next_prediction_id = 1
        self.consecutive_high_error = 0
        self.latched_anomaly = 0
        self.off_ambient_values = deque(
            maxlen=int(checkpoint.get("last_off_ambient_window", LAST_OFF_AMBIENT_WINDOW))
        )

    def current_last_off_ambient_median(self, record):
        if not self.off_ambient_values:
            if record["light_state"] == 0:
                return record["ambient_light"]
            return self.scalers["last_off_ambient_median"].mean
        values = sorted(self.off_ambient_values)
        return values[len(values) // 2]

    def add_record(self, record):
        record = dict(record)
        record["last_off_ambient_median"] = self.current_last_off_ambient_median(record)
        self.records.append(record)
        current_index = len(self.records) - 1
        results = []

        results.extend(self.score_ready_predictions(current_index))
        start_result = self.start_prediction(current_index)
        if start_result is not None:
            results.append(start_result)
        elif not results and len(self.records) < self.input_window:
            results.append({
                "time": record["time"],
                "status": "warming_up",
                "history": len(self.records),
                "required_history": self.input_window,
            })
        if record["light_state"] == 0:
            self.off_ambient_values.append(record["ambient_light"])
        return results

    def input_features_for_current_index(self, current_index):
        input_start = current_index - self.input_window + 1
        if input_start < 0:
            return None, None
        input_records = self.records[input_start:current_index + 1]
        if input_start > 0:
            feature_slice = self.records[input_start - 1:current_index + 1]
            x_features = build_features(feature_slice, self.scalers)[1:]
        else:
            x_features = build_features(input_records, self.scalers)
        return input_records, x_features

    def start_prediction(self, current_index):
        input_records, x_features = self.input_features_for_current_index(current_index)
        if x_features is None:
            return None
        current_record = self.records[current_index]
        x = torch.tensor([x_features], dtype=torch.float32, device=self.device)
        context = torch.tensor([build_context(current_record, self.scalers)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model(x, context).view(1, len(PHYSICAL_NODES))
        prediction_id = self.next_prediction_id
        self.next_prediction_id += 1
        metadata = {
            "prediction_id": prediction_id,
            "prediction_time": current_record["time"],
            "start_index": current_index,
            "current_state": global_state(current_record),
            "input_transition_count": transition_count(input_records),
            "seconds_since_last_transition": seconds_since_last_transition(input_records),
            "prediction": prediction[0].detach().cpu().tolist(),
        }
        self.pending_predictions.append(metadata)
        return {
            "time": current_record["time"],
            "status": "prediction_started",
            "prediction_id": prediction_id,
            "prediction_time": current_record["time"],
            "history": len(self.records),
            "required_history": self.input_window,
            "current_state": metadata["current_state"],
            "light_state": current_record["light_state"],
            "fan_state": current_record["fan_state"],
            "input_transition_count": metadata["input_transition_count"],
            "seconds_since_last_transition": metadata["seconds_since_last_transition"],
        }

    def score_ready_predictions(self, current_index):
        results = []
        still_pending = []
        for pending in self.pending_predictions:
            if current_index >= pending["start_index"] + self.horizon:
                results.append(self.score_prediction(pending))
            else:
                still_pending.append(pending)
        self.pending_predictions = still_pending
        return results

    def score_prediction(self, pending):
        start = pending["start_index"]
        future_slice = self.records[start:start + self.horizon + 1]
        previous_record = future_slice[0]
        target_records = future_slice[1:]
        final_record = target_records[-1]
        if target_has_state_change(previous_record, target_records):
            self.consecutive_high_error = 0
            return {
                "time": final_record["time"],
                "status": "prediction_skipped_future_transition",
                "prediction_id": pending["prediction_id"],
                "prediction_time": pending["prediction_time"],
                "target_start_time": target_records[0]["time"],
                "target_end_time": final_record["time"],
                "skipped_reason": "global_state changed inside prediction horizon",
                "current_state": pending["current_state"],
                "light_state": final_record["light_state"],
                "fan_state": final_record["fan_state"],
                "input_transition_count": pending["input_transition_count"],
                "seconds_since_last_transition": pending["seconds_since_last_transition"],
            }

        raw_target_response = build_response_target(previous_record, target_records)
        target = torch.tensor(
            [transform_response(raw_target_response, self.response_scalers)],
            dtype=torch.float32,
            device=self.device,
        )
        prediction = torch.tensor([pending["prediction"]], dtype=torch.float32, device=self.device)
        weights = torch.tensor(
            [[self.target_weights[name] for name in PHYSICAL_NODES]],
            dtype=torch.float32,
            device=self.device,
        )
        per_target_errors = (prediction - target) ** 2 * weights
        sample_errors = per_target_errors.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
        error = sample_errors[0].item()
        target_errors = {
            name: per_target_errors[0, index].item()
            for index, name in enumerate(PHYSICAL_NODES)
        }
        total_high_error = int(error > self.threshold)
        ambient_threshold = self.target_thresholds.get("ambient_light")
        power_threshold = self.target_thresholds.get("total_power_w")
        ambient_high_error = int(
            ambient_threshold is not None
            and target_errors["ambient_light"] > ambient_threshold
        )
        power_high_error = int(
            power_threshold is not None
            and target_errors["total_power_w"] > power_threshold
        )
        high_error = int(total_high_error or ambient_high_error or power_high_error)
        if high_error:
            self.consecutive_high_error += 1
        else:
            self.consecutive_high_error = 0
        current_anomaly = int(self.consecutive_high_error >= self.args.consecutive)
        if self.args.no_latch:
            anomaly = current_anomaly
        else:
            if current_anomaly:
                self.latched_anomaly = 1
            anomaly = self.latched_anomaly

        raw_prediction_response = inverse_transform_response(
            pending["prediction"],
            self.response_scalers,
        )
        raw_actual_response = inverse_transform_response(
            target[0].detach().cpu().tolist(),
            self.response_scalers,
        )
        predicted_response = {}
        actual_response = {}
        predicted_tail_values = {}
        actual_tail_values = {}
        for index, name in enumerate(PHYSICAL_NODES):
            predicted_response[name] = raw_prediction_response[index]
            actual_response[name] = raw_actual_response[index]
            predicted_tail_values[name] = previous_record[name] + predicted_response[name]
            actual_tail_values[name] = previous_record[name] + actual_response[name]

        result = {
            "time": final_record["time"],
            "status": "prediction_scored",
            "prediction_id": pending["prediction_id"],
            "prediction_time": pending["prediction_time"],
            "target_start_time": target_records[0]["time"],
            "target_end_time": final_record["time"],
            "anomaly": anomaly,
            "high_error": high_error,
            "total_high_error": total_high_error,
            "ambient_high_error": ambient_high_error,
            "power_high_error": power_high_error,
            "consecutive_high_error": self.consecutive_high_error,
            "error": error,
            "threshold": self.threshold,
            "current_state": pending["current_state"],
            "light_state": final_record["light_state"],
            "fan_state": final_record["fan_state"],
            "input_transition_count": pending["input_transition_count"],
            "seconds_since_last_transition": pending["seconds_since_last_transition"],
            "err_ambient_light": target_errors["ambient_light"],
            "err_total_power_w": target_errors["total_power_w"],
            "th_ambient_light": self.target_thresholds.get("ambient_light"),
            "th_total_power_w": self.target_thresholds.get("total_power_w"),
            "last_off_ambient_median": previous_record["last_off_ambient_median"],
            "light_age_seconds": final_record["light_age_seconds"],
            "dreo_age_seconds": final_record["dreo_age_seconds"],
            "tapo_age_seconds": final_record["tapo_age_seconds"],
            "arduino_age_seconds": final_record["arduino_age_seconds"],
        }
        for name in PHYSICAL_NODES:
            result[f"actual_{name}"] = final_record[name]
            result[f"pred_{name}"] = predicted_tail_values[name]
            result[f"actual_delta_{name}"] = actual_response[name]
            result[f"pred_delta_{name}"] = predicted_response[name]
        return result


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[INFO] Connected to MQTT broker")
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            print(f"[INFO] Subscribed: {topic}")
    else:
        print(f"[ERROR] MQTT connect failed rc={rc}")


def on_message(client, userdata, msg):
    payload_str = msg.payload.decode("utf-8", errors="ignore")
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return

    now_ts = time.time()
    if msg.topic == "iot/arduino":
        latest_state["arduino"].update(
            recv_ts=now_ts,
            ambient_light=payload.get("ambient_light"),
        )
    elif msg.topic == "iot/light_state":
        latest_state["light"].update(
            recv_ts=now_ts,
            light_state=payload.get("light_state"),
        )
    elif msg.topic == "iot/tapo_p110":
        latest_state["tapo"].update(
            recv_ts=now_ts,
            total_power_w=payload.get("power_w"),
        )
    elif msg.topic == "iot/dreo_fan":
        latest_state["dreo"].update(
            recv_ts=now_ts,
            fan_state=payload.get("fan_state"),
        )


def write_log(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as file:
            first_line = file.readline().strip()
        expected_header = ",".join(LOG_FIELDS)
        if first_line != expected_header:
            backup_path = path.with_suffix(f".{int(time.time())}.bak.csv")
            path.rename(backup_path)
            print(f"[INFO] Existing online log moved to {backup_path}")

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Online rolling multi-step GAT-GRU IDS without temperature.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--mqtt-broker", default=MQTT_BROKER)
    parser.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    parser.add_argument("--alert-topic", default=ALERT_TOPIC)
    parser.add_argument("--online-log", default=str(ONLINE_LOG))
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument(
        "--no-latch",
        action="store_true",
        help="Do not keep anomaly=1 after the current high-error streak clears.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "rolling_multistep_gat_gru_delta_predictor":
        raise SystemExit("This script expects a model trained by train_rolling_multistep_gnn_gru_ids.py")
    ids = OnlineRollingMultiStepIDS(checkpoint, args)
    log_path = Path(args.online_log)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.mqtt_broker, args.mqtt_port, 60)
    client.loop_start()

    print(f"[INFO] Model: {args.model_path}")
    print(f"[INFO] Input window: {ids.input_window}s")
    print(f"[INFO] Prediction horizon: {ids.horizon}s")
    print(f"[INFO] Last-off ambient window: {ids.off_ambient_values.maxlen}")
    print(f"[INFO] Target space: {checkpoint.get('target_space', 'raw_response')}")
    print("[INFO] Response scalers:")
    for name, scaler in ids.response_scalers.items():
        print(f"  {name}: mean={scaler.mean:.6f}, std={scaler.std:.6f}")
    print(f"[INFO] Error method: tail_median_response start_step={response_tail_start(ids.horizon) + 1}")
    print(f"[INFO] Threshold: {ids.threshold:.6f}")
    print(f"[INFO] Publishing alerts to: {args.alert_topic}")

    try:
        while True:
            record = build_current_record()
            if record is not None:
                for result in ids.add_record(record):
                    client.publish(args.alert_topic, json.dumps(result, ensure_ascii=False))
                    write_log(result, log_path)
                    status = result.get("status")
                    if status == "prediction_scored":
                        print(
                            "[ROLLING_MULTI_IDS] "
                            f"pred={result['prediction_id']} "
                            f"anomaly={result['anomaly']} "
                            f"high_error={result['high_error']} "
                            f"total_high={result['total_high_error']} "
                            f"ambient_high={result['ambient_high_error']} "
                            f"power_high={result['power_high_error']} "
                            f"error={result['error']:.6f} "
                            f"threshold={result['threshold']:.6f} "
                            f"err_amb={result['err_ambient_light']:.6f} "
                            f"th_amb={result['th_ambient_light']:.6f} "
                            f"err_pow={result['err_total_power_w']:.6f} "
                            f"th_pow={result['th_total_power_w']:.6f} "
                            f"state={result['current_state']} "
                            f"transitions={result['input_transition_count']}"
                        )
                    elif status == "prediction_skipped_future_transition":
                        print(
                            "[ROLLING_MULTI_IDS] "
                            f"pred={result['prediction_id']} skipped future transition"
                        )
                    elif status == "warming_up":
                        print(
                            "[ROLLING_MULTI_IDS] "
                            f"warming_up={result['history']}/{result['required_history']}"
                        )
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
