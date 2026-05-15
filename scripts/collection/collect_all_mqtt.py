import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt


MQTT_BROKER = "172.24.224.223"
MQTT_PORT = 1883

MQTT_TOPICS = [
    "iot/arduino",
    "iot/light_state",
    "iot/tapo_p110",
    "iot/dreo_fan",
]

OUTPUT_INTERVAL = 1.0
COLLECT_DURATION = 3600 * 24
CSV_FILENAME = "aligned_all_data_clean_delay.csv"

ARDUINO_TTL = 5.0
LIGHT_TTL = 10.0
TAPO_TTL = 5.0
DREO_TTL = 35.0


FIELDNAMES = [
    "time",
    "temperature",
    "humidity",
    "ambient_light",
    "light_state",
    "fan_power_w",
    "fan_state",
    "arduino_age_seconds",
    "light_age_seconds",
    "tapo_age_seconds",
    "dreo_age_seconds",
    "arduino_valid",
    "light_valid",
    "tapo_valid",
    "dreo_valid",
]


latest_state = {
    "arduino": {
        "recv_ts": None,
        "temperature": None,
        "humidity": None,
        "ambient_light": None,
    },
    "light": {
        "recv_ts": None,
        "light_state": None,
    },
    "tapo": {
        "recv_ts": None,
        "power_w": None,
    },
    "dreo": {
        "recv_ts": None,
        "fan_state": None,
    },
}


def iso_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def age_seconds(recv_ts):
    if recv_ts is None:
        return None
    return round(time.time() - recv_ts, 3)


def is_fresh(recv_ts, ttl):
    return recv_ts is not None and (time.time() - recv_ts) <= ttl


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[INFO] Connected to broker")
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            print(f"[INFO] Subscribed to: {topic}")
    else:
        print(f"[ERROR] Failed to connect, rc={rc}")


def on_message(client, userdata, msg):
    payload_str = msg.payload.decode("utf-8", errors="ignore")
    print(f"[RECV] topic={msg.topic} payload={payload_str}")

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        print("[WARN] Invalid JSON, skipped")
        return

    now_ts = time.time()

    if msg.topic == "iot/arduino":
        latest_state["arduino"].update(
            recv_ts=now_ts,
            temperature=payload.get("temperature"),
            humidity=payload.get("humidity"),
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
            power_w=payload.get("power_w"),
        )

    elif msg.topic == "iot/dreo_fan":
        latest_state["dreo"].update(
            recv_ts=now_ts,
            fan_state=payload.get("fan_state"),
        )


def build_aligned_record():
    arduino_valid = is_fresh(latest_state["arduino"]["recv_ts"], ARDUINO_TTL)
    light_valid = is_fresh(latest_state["light"]["recv_ts"], LIGHT_TTL)
    tapo_valid = is_fresh(latest_state["tapo"]["recv_ts"], TAPO_TTL)
    dreo_valid = is_fresh(latest_state["dreo"]["recv_ts"], DREO_TTL)

    record = {field: None for field in FIELDNAMES}
    record.update(
        time=iso_utc_now(),
        arduino_age_seconds=age_seconds(latest_state["arduino"]["recv_ts"]),
        light_age_seconds=age_seconds(latest_state["light"]["recv_ts"]),
        tapo_age_seconds=age_seconds(latest_state["tapo"]["recv_ts"]),
        dreo_age_seconds=age_seconds(latest_state["dreo"]["recv_ts"]),
        arduino_valid=1 if arduino_valid else 0,
        light_valid=1 if light_valid else 0,
        tapo_valid=1 if tapo_valid else 0,
        dreo_valid=1 if dreo_valid else 0,
    )

    if arduino_valid:
        record.update(
            temperature=latest_state["arduino"]["temperature"],
            humidity=latest_state["arduino"]["humidity"],
            ambient_light=latest_state["arduino"]["ambient_light"],
        )

    if light_valid:
        record["light_state"] = latest_state["light"]["light_state"]

    if tapo_valid:
        record["fan_power_w"] = latest_state["tapo"]["power_w"]

    if dreo_valid:
        record["fan_state"] = latest_state["dreo"]["fan_state"]

    return record


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[INFO] Connecting to broker {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    start_time = time.time()
    csv_path = Path(CSV_FILENAME)
    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()

        try:
            while True:
                if time.time() - start_time >= COLLECT_DURATION:
                    print("[INFO] Collection finished.")
                    break

                record = build_aligned_record()
                print("[ALIGNED]", json.dumps(record, ensure_ascii=False))
                writer.writerow(record)
                f.flush()
                time.sleep(OUTPUT_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            client.loop_stop()
            client.disconnect()
            print(f"[INFO] Data saved to {CSV_FILENAME}")


if __name__ == "__main__":
    main()
