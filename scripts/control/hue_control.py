import argparse
import json
import os
import random
import time

import requests


HUE_BRIDGE_IP = "PHILIPS_HUE_BRIDGE_IP"
HUE_USERNAME = "YOUR_HUE_USERNAME"
HUE_LIGHT_ID = "3"
# loop --random --min-interval 90 --max-interval 180 --count 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Control a Philips Hue light locally through the Hue Bridge REST API."
    )
    parser.add_argument(
        "action",
        choices=["on", "off", "toggle", "blink", "loop", "status"],
        help="Action to run. Use loop to repeatedly switch on/off for IDS testing.",
    )
    parser.add_argument(
        "--bridge-ip",
        default=os.getenv("HUE_BRIDGE_IP", HUE_BRIDGE_IP),
        help="Hue Bridge IP address.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("HUE_USERNAME", HUE_USERNAME),
        help="Hue API username/key.",
    )
    parser.add_argument(
        "--light-id",
        default=os.getenv("HUE_LIGHT_ID", HUE_LIGHT_ID),
        help="Hue light id. Your current project default is 3.",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=254,
        help="Brightness when turning on, 1-254.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between on/off changes for loop/blink.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use a random interval between --min-interval and --max-interval.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=5.0,
        help="Minimum random interval in seconds. Values below 5 are raised to 5.",
    )
    parser.add_argument(
        "--max-interval",
        type=float,
        default=20.0,
        help="Maximum random interval in seconds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of state changes for loop/blink. Use 0 for infinite.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def next_sleep_seconds(args):
    if not args.random:
        return max(5.0, args.interval)

    minimum = max(5.0, args.min_interval)
    maximum = max(minimum, args.max_interval)
    return random.uniform(minimum, maximum)


def hue_url(args, suffix=""):
    return f"http://{args.bridge_ip}/api/{args.username}/lights/{args.light_id}{suffix}"


def check_hue_response(data):
    if isinstance(data, list):
        errors = [item["error"] for item in data if "error" in item]
        if errors:
            raise RuntimeError(json.dumps(errors, ensure_ascii=False))
    return data


def get_state(args):
    response = requests.get(hue_url(args), timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
    state = data.get("state", {})
    return {
        "on": bool(state.get("on")),
        "bri": state.get("bri"),
        "reachable": state.get("reachable"),
        "name": data.get("name"),
    }


def set_state(args, is_on):
    body = {"on": bool(is_on)}
    if is_on:
        body["bri"] = max(1, min(254, args.brightness))

    response = requests.put(hue_url(args, "/state"), json=body, timeout=args.timeout)
    response.raise_for_status()
    check_hue_response(response.json())
    print(f"[HUE] set light {args.light_id} on={bool(is_on)} bri={body.get('bri', '-')}")


def run_loop(args):
    if args.count < 0:
        raise ValueError("--count must be >= 0")

    current = False
    changes = 0
    print(
        f"[HUE] loop start: light={args.light_id}, interval={args.interval}s, "
        f"random={args.random}, count={'infinite' if args.count == 0 else args.count}"
    )
    while args.count == 0 or changes < args.count:
        current = not current
        set_state(args, current)
        changes += 1
        sleep_seconds = next_sleep_seconds(args)
        print(f"[HUE] wait {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)


def run_blink(args):
    if args.count < 0:
        raise ValueError("--count must be >= 0")

    cycles = 0
    print(
        f"[HUE] blink start: light={args.light_id}, interval={args.interval}s, "
        f"random={args.random}, cycles={'infinite' if args.count == 0 else args.count}"
    )
    while args.count == 0 or cycles < args.count:
        set_state(args, True)
        sleep_seconds = next_sleep_seconds(args)
        print(f"[HUE] wait {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)
        set_state(args, False)
        sleep_seconds = next_sleep_seconds(args)
        print(f"[HUE] wait {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)
        cycles += 1


def main():
    args = parse_args()

    if args.action == "status":
        state = get_state(args)
        print(json.dumps(state, ensure_ascii=False))
    elif args.action == "on":
        set_state(args, True)
    elif args.action == "off":
        set_state(args, False)
    elif args.action == "toggle":
        state = get_state(args)
        set_state(args, not state["on"])
    elif args.action == "loop":
        run_loop(args)
    elif args.action == "blink":
        run_blink(args)


if __name__ == "__main__":
    main()

# loop --random --min-interval 8 --max-interval 20 --count 0
