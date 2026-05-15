# Data Collection Scripts

This folder contains scripts used to collect and align IoT telemetry for the contextual IDS experiments.

## Scripts

| Script | Purpose |
|---|---|
| `read_arduino.py` | Read Arduino serial sensor values and publish temperature/humidity/ambient light to MQTT. |
| `philips_hue_to_mqtt.py` | Poll Philips Hue Bridge locally and publish `light_state` to MQTT. |
| `tapo_p110_to_mqtt.py` | Poll Tapo P110 power data and publish `power_w` to MQTT. |
| `dreo_ha_to_mqtt.py` | Poll Dreo fan state through Home Assistant and publish `fan_state` to MQTT. |
| `collect_all_mqtt.py` | Subscribe to MQTT topics and write aligned CSV data for training. |

## Configuration

Sensitive values are intentionally not committed. Set them with environment variables or command-line arguments.

Common variables:

```powershell
$env:MQTT_BROKER="172.24.224.223"
$env:HUE_BRIDGE_IP="192.168.88.xxx"
$env:HUE_USERNAME="your_hue_username"
$env:TAPO_HOST="192.168.88.xxx"
$env:TAPO_USERNAME="your_tapo_email"
$env:TAPO_PASSWORD="your_tapo_password"
$env:HOME_ASSISTANT_URL="http://192.168.88.xxx:8123"
$env:HOME_ASSISTANT_TOKEN="your_home_assistant_long_lived_token"
```

## Example Run Order

Open separate terminals for each collector:

```powershell
python read_arduino.py
python philips_hue_to_mqtt.py
python tapo_p110_to_mqtt.py
python dreo_ha_to_mqtt.py
python collect_all_mqtt.py
```
