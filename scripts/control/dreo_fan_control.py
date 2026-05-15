import argparse
import json
import os
import random
import time

import requests
# loop --random --min-interval 40 --max-interval 120 --count 0 --percentage 50
# loop --random --min-interval 90 --max-interval 180 --count 0
try:
    from dreo_ha_to_mqtt import (
        DREO_FAN_ENTITY_ID,
        HOME_ASSISTANT_TOKEN,
        HOME_ASSISTANT_URL,
    )
except ImportError:
    HOME_ASSISTANT_URL = "http://HOME_ASSISTANT_IP:8123"
    HOME_ASSISTANT_TOKEN = "YOUR_HOME_ASSISTANT_LONG_LIVED_ACCESS_TOKEN"
    DREO_FAN_ENTITY_ID = "fan.air_circulator"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the Dreo fan through Home Assistant for IDS data collection."
    )
    parser.add_argument(
        "action",
        choices=["on", "off", "toggle", "loop", "status"],
        help="Action to run. Use loop for repeated on/off switching.",
    )
    parser.add_argument(
        "--ha-url",
        default=os.getenv("HOME_ASSISTANT_URL", HOME_ASSISTANT_URL),
        help="Home Assistant base URL.",
    )
    parser.add_argument(
        "--ha-token",
        default=os.getenv("HOME_ASSISTANT_TOKEN", HOME_ASSISTANT_TOKEN),
        help="Home Assistant long-lived access token.",
    )
    parser.add_argument(
        "--entity-id",
        default=os.getenv("DREO_FAN_ENTITY_ID", DREO_FAN_ENTITY_ID),
        help="Dreo fan entity id in Home Assistant.",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=50,
        help="Fan speed percentage when turning on.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Fixed seconds between state changes in loop mode.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random interval between --min-interval and --max-interval.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=40.0,
        help="Minimum random interval in seconds.",
    )
    parser.add_argument(
        "--max-interval",
        type=float,
        default=120.0,
        help="Maximum random interval in seconds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of state changes in loop mode. Use 0 for infinite.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def check_config(args):
    if not args.ha_token or args.ha_token == "YOUR_HOME_ASSISTANT_LONG_LIVED_ACCESS_TOKEN":
        raise SystemExit(
            "Missing Home Assistant token. Set HOME_ASSISTANT_TOKEN at the top "
            "of this file or pass --ha-token."
        )


def headers(args):
    return {
        "Authorization": f"Bearer {args.ha_token}",
        "Content-Type": "application/json",
    }


def ha_url(args, path):
    return f"{args.ha_url.rstrip('/')}{path}"


def get_state(args):
    response = requests.get(
        ha_url(args, f"/api/states/{args.entity_id}"),
        headers=headers(args),
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()
    state = data.get("state")
    attrs = data.get("attributes", {})
    return {
        "state": state,
        "is_on": state == "on",
        "percentage": attrs.get("percentage"),
        "preset_mode": attrs.get("preset_mode"),
        "oscillating": attrs.get("oscillating"),
        "last_changed": data.get("last_changed"),
        "last_updated": data.get("last_updated"),
    }


def call_service(args, service, payload=None):
    body = {"entity_id": args.entity_id}
    if payload:
        body.update(payload)

    response = requests.post(
        ha_url(args, f"/api/services/fan/{service}"),
        headers=headers(args),
        json=body,
        timeout=args.timeout,
    )
    response.raise_for_status()
    print(f"[HA] fan.{service} {json.dumps(body, ensure_ascii=False)}")


def turn_on(args):
    percentage = max(1, min(100, args.percentage))
    call_service(args, "turn_on", {"percentage": percentage})


def turn_off(args):
    call_service(args, "turn_off")


def toggle(args):
    state = get_state(args)
    if state["is_on"]:
        turn_off(args)
    else:
        turn_on(args)


def next_sleep_seconds(args):
    if not args.random:
        return max(10.0, args.interval)

    minimum = max(10.0, args.min_interval)
    maximum = max(minimum, args.max_interval)
    return random.uniform(minimum, maximum)


def run_loop(args):
    if args.count < 0:
        raise ValueError("--count must be >= 0")

    state = get_state(args)
    current_on = state["is_on"]
    changes = 0

    print(
        f"[DREO] loop start: entity={args.entity_id}, "
        f"current={state['state']}, random={args.random}, "
        f"count={'infinite' if args.count == 0 else args.count}"
    )

    while args.count == 0 or changes < args.count:
        current_on = not current_on

        if current_on:
            turn_on(args)
        else:
            turn_off(args)

        changes += 1
        sleep_seconds = next_sleep_seconds(args)
        print(f"[DREO] wait {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)


def main():
    args = parse_args()
    check_config(args)

    if args.action == "status":
        print(json.dumps(get_state(args), ensure_ascii=False, indent=2))
    elif args.action == "on":
        turn_on(args)
    elif args.action == "off":
        turn_off(args)
    elif args.action == "toggle":
        toggle(args)
    elif args.action == "loop":
        run_loop(args)


if __name__ == "__main__":
    main()
