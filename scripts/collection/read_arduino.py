# import json
# import time
# import serial
# import paho.mqtt.client as mqtt
#
# # ========= 配置区 =========
# SERIAL_PORT = "COM3"          # 改成你的 Arduino 端口，比如 COM3 / COM4
# BAUD_RATE = 9600
#
# MQTT_BROKER = "172.24.224.223"   # 你的 Linux broker IP
# MQTT_PORT = 1883
# MQTT_TOPIC = "iot/arduino"
#
# SERIAL_STARTUP_DELAY = 2
#
#
# def main():
#     print(f"[INFO] Opening serial port: {SERIAL_PORT}")
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#     time.sleep(SERIAL_STARTUP_DELAY)
#
#     print(f"[INFO] Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
#     client = mqtt.Client()
#     client.connect(MQTT_BROKER, MQTT_PORT, 60)
#
#     while True:
#         try:
#             line = ser.readline().decode("utf-8", errors="ignore").strip()
#             if not line:
#                 continue
#
#             print("[SERIAL]", line)
#
#             parts = line.split(",")
#             if len(parts) != 3:
#                 print("[WARN] Bad format, skipped")
#                 continue
#
#             temperature = float(parts[0])
#             humidity = float(parts[1])
#             ambient_light = int(parts[2])
#
#             payload = {
#                 "timestamp": time.time(),
#                 "temperature": temperature,
#                 "humidity": humidity,
#                 "ambient_light": ambient_light
#             }
#
#             message = json.dumps(payload)
#             client.publish(MQTT_TOPIC, message)
#             print("[MQTT] Published:", message)
#
#         except KeyboardInterrupt:
#             print("Stopped by user")
#             break
#         except Exception as e:
#             print("[ERROR]", e)
#             time.sleep(1)
#
#     ser.close()
#     client.disconnect()
#
#
# if __name__ == "__main__":
#     main()
# import json
# import time
# import serial
# import paho.mqtt.client as mqtt
# from datetime import datetime, timezone
#
# # ========= 配置区 =========
# SERIAL_PORT = "COM3"
# BAUD_RATE = 9600
#
# MQTT_BROKER = "172.24.224.223"
# MQTT_PORT = 1883
# MQTT_TOPIC = "iot/arduino"
#
# SERIAL_STARTUP_DELAY = 2
#
#
# def iso_utc_now():
#     return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
#
#
# def main():
#     print(f"[INFO] Opening serial port: {SERIAL_PORT}")
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#     time.sleep(SERIAL_STARTUP_DELAY)
#
#     print(f"[INFO] Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
#     client = mqtt.Client()
#     client.connect(MQTT_BROKER, MQTT_PORT, 60)
#
#     while True:
#         try:
#             line = ser.readline().decode("utf-8", errors="ignore").strip()
#             if not line:
#                 continue
#
#             print("[SERIAL]", line)
#
#             parts = line.split(",")
#             if len(parts) != 3:
#                 print("[WARN] Bad format, skipped")
#                 continue
#
#             temperature = float(parts[0])
#             humidity = float(parts[1])
#             ambient_light = int(parts[2])
#
#             payload = {
#                 "time": iso_utc_now(),
#                 "temperature": temperature,
#                 "humidity": humidity,
#                 "ambient_light": ambient_light
#             }
#
#             message = json.dumps(payload, ensure_ascii=False)
#             client.publish(MQTT_TOPIC, message)
#             print("[MQTT] Published:", message)
#
#         except KeyboardInterrupt:
#             print("Stopped by user")
#             break
#         except Exception as e:
#             print("[ERROR]", e)
#             time.sleep(1)
#
#     ser.close()
#     client.disconnect()
#
#
# if __name__ == "__main__":
#     main()
import json
import time
import serial
import paho.mqtt.client as mqtt
from datetime import datetime, timezone

# ========= 配置区 =========
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

MQTT_BROKER = "172.24.224.223"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/arduino"

SERIAL_STARTUP_DELAY = 2


def iso_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    print(f"[INFO] Opening serial port: {SERIAL_PORT}")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(SERIAL_STARTUP_DELAY)

    print(f"[INFO] Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print("[SERIAL]", line)

            parts = line.split(",")
            if len(parts) != 3:
                print("[WARN] Bad format, skipped")
                continue

            temperature = float(parts[0])
            humidity = float(parts[1])
            ambient_light = int(parts[2])

            payload = {
                "time": iso_utc_now(),
                "temperature": temperature,
                "humidity": humidity,
                "ambient_light": ambient_light
            }

            message = json.dumps(payload, ensure_ascii=False)
            client.publish(MQTT_TOPIC, message)
            print("[MQTT] Published:", message)

        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(1)

    ser.close()
    client.disconnect()


if __name__ == "__main__":
    main()
