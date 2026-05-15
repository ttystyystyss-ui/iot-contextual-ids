import argparse
import json
import os
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import requests


DEFAULT_MQTT_BROKER = "172.24.224.223"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_TOPIC = "iot/light_state"

HUE_BRIDGE_IP = "PHILIPS_HUE_BRIDGE_IP"
HUE_USERNAME = "YOUR_HUE_USERNAME"
HUE_LIGHT_ID = "3"
HUE_POLL_INTERVAL = 1.0
PUBLISH_EVERY_POLL = True


def iso_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a Philips Hue Bridge locally and publish light state to MQTT."
    )
    parser.add_argument(
        "--bridge-ip",
        default=os.getenv("HUE_BRIDGE_IP", HUE_BRIDGE_IP),
        help="Philips Hue Bridge IP address. Can also use HUE_BRIDGE_IP.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("HUE_USERNAME", HUE_USERNAME),
        help="Hue API username/key. Can also use HUE_USERNAME.",
    )
    parser.add_argument(
        "--light-id",
        default=os.getenv("HUE_LIGHT_ID", HUE_LIGHT_ID),
        help="Hue light id to read. Can also use HUE_LIGHT_ID.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("HUE_POLL_INTERVAL", str(HUE_POLL_INTERVAL))),
        help="Seconds between local Hue status reads.",
    )
    parser.add_argument(
        "--mqtt-broker",
        default=os.getenv("MQTT_BROKER", DEFAULT_MQTT_BROKER),
        help="MQTT broker host.",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=int(os.getenv("MQTT_PORT", str(DEFAULT_MQTT_PORT))),
        help="MQTT broker port.",
    )
    parser.add_argument(
        "--mqtt-topic",
        default=os.getenv("MQTT_TOPIC", DEFAULT_MQTT_TOPIC),
        help="MQTT topic to publish light_state messages to.",
    )
    parser.add_argument(
        "--publish-every-poll",
        action="store_true",
        default=PUBLISH_EVERY_POLL,
        help="Publish every poll instead of only when the state changes.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Create a Hue username. Press the Hue Bridge button before running.",
    )
    parser.add_argument(
        "--list-lights",
        action="store_true",
        help="List all Hue lights and their current on/off state, then exit.",
    )
    return parser.parse_args()


def register_username(bridge_ip):
    url = f"http://{bridge_ip}/api"
    response = requests.post(
        url,
        json={"devicetype": "pythonproject#hue_to_mqtt"},
        timeout=5,
    )
    response.raise_for_status()
    result = response.json()

    if not result:
        raise RuntimeError("Hue Bridge returned an empty response.")

    item = result[0]
    if "success" in item:
        username = item["success"]["username"]
        print(f"[INFO] Hue username created: {username}")
        print("[INFO] Use it with: --username " + username)
        return username

    error = item.get("error", {})
    description = error.get("description", "unknown error")
    raise RuntimeError(f"Hue registration failed: {description}")


def read_light_state(bridge_ip, username, light_id):
    url = f"http://{bridge_ip}/api/{username}/lights/{light_id}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and data and "error" in data[0]:
        raise RuntimeError(data[0]["error"].get("description", "Hue API error"))

    state = data["state"]
    return {
        "reachable": bool(state.get("reachable", False)),
        "light_state": 1 if state.get("on") else 0,
        "raw_on": bool(state.get("on")),
        "name": data.get("name"),
    }


def read_all_lights(bridge_ip, username):
    url = f"http://{bridge_ip}/api/{username}/lights"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and data and "error" in data[0]:
        raise RuntimeError(data[0]["error"].get("description", "Hue API error"))

    return data


def print_all_lights(bridge_ip, username):
    lights = read_all_lights(bridge_ip, username)
    if not lights:
        print("[INFO] No lights found.")
        return

    for light_id, light in lights.items():
        state = light.get("state", {})
        is_on = bool(state.get("on", False))
        reachable = bool(state.get("reachable", False))
        brightness = state.get("bri")
        print(
            f"id={light_id} | name={light.get('name')} | "
            f"on={is_on} | light_state={1 if is_on else 0} | "
            f"reachable={reachable} | bri={brightness}"
        )


def publish_state(client, topic, state):
    payload = {
        "time": iso_utc_now(),
        "light_state": state["light_state"],
        "reachable": state["reachable"],
        "source": "philips_hue_bridge",
        "light_name": state["name"],
    }
    message = json.dumps(payload, ensure_ascii=False)
    client.publish(topic, message)
    print(f"[MQTT] {topic} {message}")


def main():
    args = parse_args()

    if not args.bridge_ip:
        raise SystemExit("Missing Hue Bridge IP. Use --bridge-ip or HUE_BRIDGE_IP.")

    if args.bridge_ip == "PHILIPS-HUE-IP":
        raise SystemExit("Please set HUE_BRIDGE_IP at the top of this file first.")

    if args.register:
        register_username(args.bridge_ip)
        return

    if not args.username:
        raise SystemExit(
            "Missing Hue username. Press the Hue Bridge button, then run: "
            "python philips_hue_to_mqtt.py --bridge-ip <ip> --register"
        )

    if args.list_lights:
        print_all_lights(args.bridge_ip, args.username)
        return

    client = mqtt.Client()
    client.connect(args.mqtt_broker, args.mqtt_port, 60)
    print(f"[INFO] Connected to MQTT broker: {args.mqtt_broker}:{args.mqtt_port}")
    print(f"[INFO] Reading Hue light {args.light_id} from bridge {args.bridge_ip}")

    last_state = None
    while True:
        try:
            state = read_light_state(args.bridge_ip, args.username, args.light_id)
            current_state = state["light_state"]

            if args.publish_every_poll or current_state != last_state:
                publish_state(client, args.mqtt_topic, state)
                last_state = current_state
            else:
                print(
                    f"[INFO] unchanged light_state={current_state} "
                    f"reachable={state['reachable']}"
                )

        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as exc:
            print(f"[ERROR] {exc}")

        time.sleep(args.poll_interval)

    client.disconnect()


if __name__ == "__main__":
    main()
