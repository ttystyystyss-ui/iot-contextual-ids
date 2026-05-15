# Device Control Scripts

This folder contains scripts used to automatically switch device states during data collection and IDS testing.

## Scripts

| Script | Purpose |
|---|---|
| `hue_control.py` | Control Philips Hue light through the local Hue Bridge REST API. |
| `dreo_fan_control.py` | Control Dreo fan through Home Assistant service calls. |

## Configuration

Sensitive values are intentionally not committed. Set them with environment variables or command-line arguments.

```powershell
$env:HUE_BRIDGE_IP="192.168.88.xxx"
$env:HUE_USERNAME="your_hue_username"
$env:HOME_ASSISTANT_URL="http://192.168.88.xxx:8123"
$env:HOME_ASSISTANT_TOKEN="your_home_assistant_long_lived_token"
```

## Examples

Hue light loop with random interval:

```powershell
python hue_control.py loop --random --min-interval 90 --max-interval 180 --count 0
```

Dreo fan loop with random interval:

```powershell
python dreo_fan_control.py loop --random --min-interval 90 --max-interval 180 --count 0 --percentage 50
```
