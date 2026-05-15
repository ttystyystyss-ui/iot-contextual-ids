import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

try:
    from kasa import Credentials, Discover
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'python-kasa'. Run: python -m pip install python-kasa"
    ) from exc


DEFAULT_TAPO_HOST = "TAPO_P110_IP"
DEFAULT_TAPO_USERNAME = "YOUR_TAPO_EMAIL"
DEFAULT_TAPO_PASSWORD = "YOUR_TAPO_PASSWORD"
DEFAULT_POLL_INTERVAL = 1.0

DEFAULT_MQTT_BROKER = "172.24.224.223"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_TOPIC = "iot/tapo_p110"


def iso_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a Tapo P110 locally and publish plug/energy data to MQTT."
    )
    parser.add_argument(
        "--host",
        "--ip",
        dest="host",
        default=os.getenv("TAPO_HOST", os.getenv("TAPO_IP", DEFAULT_TAPO_HOST)),
        help="Tapo P110 IP address.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("TAPO_USERNAME", DEFAULT_TAPO_USERNAME),
        help="Tapo account email. Can also use TAPO_USERNAME.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("TAPO_PASSWORD", DEFAULT_TAPO_PASSWORD),
        help="Tapo account password. Can also use TAPO_PASSWORD.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("TAPO_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))),
        help="Seconds between P110 reads.",
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
        help="MQTT topic for P110 telemetry.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Read once, print one payload, then exit.",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print all feature ids reported by python-kasa, then exit.",
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Only print readings; do not publish to MQTT.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Device communication timeout in seconds.",
    )
    return parser.parse_args()


def require_credentials(args):
    if (
        not args.username
        or not args.password
        or args.username == "YOUR_TAPO_EMAIL"
        or args.password == "YOUR_TAPO_PASSWORD"
    ):
        raise SystemExit(
            "Missing Tapo credentials. Set DEFAULT_TAPO_USERNAME and "
            "DEFAULT_TAPO_PASSWORD at the top of tapo_p110_to_mqtt.py, or use "
            "--username and --password."
        )


async def connect_device(args):
    require_credentials(args)
    credentials = Credentials(args.username, args.password)
    try:
        device = await Discover.discover_single(
            args.host,
            credentials=credentials,
            timeout=args.timeout,
            discovery_timeout=args.timeout,
        )
    except Exception as exc:
        message = str(exc)
        if "TPAP" in message or "Unsupported device" in message:
            raise SystemExit(
                "Tapo P110 was found, but local third-party access is blocked "
                "by the device firmware.\n\n"
                "Fix in Tapo app:\n"
                "1. Open Tapo app\n"
                "2. Go to Me / Third-Party Services or Tapo Lab\n"
                "3. Enable Third-Party Compatibility\n"
                "4. Wait a few seconds, then run this script again\n\n"
                f"Original error: {message}"
            ) from exc
        raise

    if device is None:
        raise SystemExit(f"No Tapo device found at {args.host}")

    await device.update()
    return device


def clean_value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def feature_value(device, *feature_ids):
    for feature_id in feature_ids:
        feature = device.features.get(feature_id)
        if feature is not None:
            try:
                return clean_value(feature.value)
            except Exception:
                return None
    return None


def all_features(device):
    features = {}
    for feature_id, feature in sorted(device.features.items()):
        try:
            value = clean_value(feature.value)
        except Exception as exc:
            value = f"<error: {exc}>"
        unit = getattr(feature, "unit", None)
        features[feature_id] = {
            "name": getattr(feature, "name", feature_id),
            "value": value,
            "unit": str(unit) if unit is not None else None,
        }
    return features


def build_payload(device):
    return {
        "time": iso_utc_now(),
        "source": "tapo_p110",
        "host": getattr(device, "host", None),
        "alias": getattr(device, "alias", None),
        "model": getattr(device, "model", None),
        "power_w": feature_value(
            device,
            "current_consumption",
            "current_power",
            "power",
        ),
    }


def print_features(device):
    print(json.dumps(all_features(device), ensure_ascii=False, indent=2))


async def main_async():
    args = parse_args()
    device = await connect_device(args)

    if args.list_features:
        await device.update()
        print_features(device)
        return

    client = None
    if not args.no_mqtt and not args.once:
        client = mqtt.Client()
        client.connect(args.mqtt_broker, args.mqtt_port, 60)
        client.loop_start()
        print(f"[INFO] Connected to MQTT broker: {args.mqtt_broker}:{args.mqtt_port}")
        print(f"[INFO] Publishing Tapo P110 data to: {args.mqtt_topic}")

    print(f"[INFO] Reading Tapo P110 at {args.host}")

    try:
        while True:
            await device.update()
            payload = build_payload(device)
            message = json.dumps(payload, ensure_ascii=False)

            if client is not None:
                client.publish(args.mqtt_topic, message)
                print(f"[MQTT] {args.mqtt_topic} {message}")
            else:
                print(message)

            if args.once:
                break

            await asyncio.sleep(max(0.2, args.poll_interval))
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()
        await device.disconnect()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
