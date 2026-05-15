import argparse
import json
import os
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import requests


HOME_ASSISTANT_URL = "http://HOME_ASSISTANT_IP:8123"
HOME_ASSISTANT_TOKEN = "YOUR_HOME_ASSISTANT_LONG_LIVED_ACCESS_TOKEN"
DREO_FAN_ENTITY_ID = "fan.air_circulator"
POLL_INTERVAL_SECONDS = 10.0

MQTT_BROKER = "172.24.224.223"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/dreo_fan"


def iso_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a Dreo fan from Home Assistant and publish state to MQTT."
    )
    parser.add_argument("--ha-url", default=os.getenv("HOME_ASSISTANT_URL", HOME_ASSISTANT_URL))
    parser.add_argument("--ha-token", default=os.getenv("HOME_ASSISTANT_TOKEN", HOME_ASSISTANT_TOKEN))
    parser.add_argument("--entity-id", default=os.getenv("DREO_FAN_ENTITY_ID", DREO_FAN_ENTITY_ID))
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("DREO_POLL_INTERVAL", str(POLL_INTERVAL_SECONDS))),
    )
    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", MQTT_BROKER))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", str(MQTT_PORT))))
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_TOPIC", MQTT_TOPIC))
    return parser.parse_args()


def check_config(args):
    missing = []
    if "HOME-ASSISTANT-IP" in args.ha_url:
        missing.append("HOME_ASSISTANT_URL")
    if args.ha_token == "YOUR_HOME_ASSISTANT_LONG_LIVED_ACCESS_TOKEN":
        missing.append("HOME_ASSISTANT_TOKEN")
    if args.entity_id == "fan.your_dreo_fan":
        missing.append("DREO_FAN_ENTITY_ID")

    if missing:
        raise SystemExit(
            "Please set these values at the top of dreo_ha_to_mqtt.py first: "
            + ", ".join(missing)
        )


def read_ha_state(ha_url, token, entity_id):
    url = f"{ha_url.rstrip('/')}/api/states/{entity_id}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def build_payload(entity_state):
    state = entity_state.get("state")
    fan_state = 1 if state == "on" else 0

    return {
        "time": iso_utc_now(),
        "source": "home_assistant_dreo",
        "entity_id": entity_state.get("entity_id"),
        "fan_state": fan_state,
        "raw_state": state,
        "last_changed": entity_state.get("last_changed"),
        "last_updated": entity_state.get("last_updated"),
    }


def main():
    args = parse_args()
    check_config(args)

    client = mqtt.Client()
    client.connect(args.mqtt_broker, args.mqtt_port, 60)
    print(f"[INFO] Connected to MQTT broker: {args.mqtt_broker}:{args.mqtt_port}")
    print(f"[INFO] Reading Home Assistant entity: {args.entity_id}")

    try:
        while True:
            try:
                entity_state = read_ha_state(args.ha_url, args.ha_token, args.entity_id)
                payload = build_payload(entity_state)
                message = json.dumps(payload, ensure_ascii=False)
                client.publish(args.mqtt_topic, message)
                print(f"[MQTT] {args.mqtt_topic} {message}")
            except Exception as exc:
                print(f"[ERROR] {exc}")

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
