from __future__ import annotations

import cmath
import math
import os
import pickle
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time


def load_machine_env_file():
    env_file_candidates = [
        os.getenv("TP2_COCHE_ENV_FILE", "").strip(),
        "~/.config/tp2/coche-jetson.env",
        "/etc/tp2/coche-jetson.env",
    ]

    for candidate in env_file_candidates:
        if not candidate:
            continue

        env_path = Path(candidate).expanduser()
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, value)
        return env_path

    return None


load_machine_env_file()

import cv2
import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
except Exception as exc:  # pragma: no cover - depends on runtime display/input stack
    pynput_keyboard = None
    PYNPUT_IMPORT_ERROR = exc
else:
    PYNPUT_IMPORT_ERROR = None

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:
    InputDevice = None
    ecodes = None
    list_devices = None

# The live car script runs on EPC, but its default inference backend is Jetson.
# Keep these as defaults only so operators can still override them from env.
os.environ.setdefault("TP2_INFERENCE_MODE", "local")
os.environ.setdefault("TP2_INFERENCE_TARGET", "workflow")
os.environ.setdefault("ROBOFLOW_LOCAL_API_URL", "http://100.115.99.8:9001")
os.environ.setdefault("ROBOFLOW_WORKSPACE", "1-v8mk1")
os.environ.setdefault("ROBOFLOW_WORKFLOW", "custom-workflow-2")

from roboflow_runtime import (
    InferenceConfig,
    create_client,
    draw_predictions_on_image,
    extract_predictions,
    infer_one_image,
    local_endpoint_reachable,
)


BIND_IP = os.getenv("TP2_BIND_IP", "172.16.0.1")
BIND_PORT = int(os.getenv("TP2_BIND_PORT", "20001"))
STEERING_CENTER = float(os.getenv("TP2_STEERING_CENTER", "0.25"))
PS4_DEADZONE = float(os.getenv("TP2_PS4_DEADZONE", "0.08"))
PS4_SCAN_INTERVAL_SEC = float(os.getenv("TP2_PS4_SCAN_INTERVAL_SEC", "1.0"))
PS4_AXIS_STEERING = int(os.getenv("TP2_PS4_AXIS_STEERING", str(ecodes.ABS_X if ecodes else 0)))
PS4_AXIS_L2 = int(os.getenv("TP2_PS4_AXIS_L2", str(ecodes.ABS_Z if ecodes else 0)))
PS4_AXIS_R2 = int(os.getenv("TP2_PS4_AXIS_R2", str(ecodes.ABS_RZ if ecodes else 0)))
DEFAULT_PS4_BUTTON_ESTOP = getattr(ecodes, "BTN_SOUTH", 304) if ecodes else 304
PS4_BUTTON_ESTOP = int(os.getenv("TP2_PS4_BUTTON_ESTOP", str(DEFAULT_PS4_BUTTON_ESTOP)))
DEFAULT_PS4_BUTTON_BOOST_REVERSE = getattr(ecodes, "BTN_TL", 310) if ecodes else 310
DEFAULT_PS4_BUTTON_BOOST_FORWARD = getattr(ecodes, "BTN_TR", 311) if ecodes else 311
PS4_BUTTON_BOOST_REVERSE = int(
    os.getenv("TP2_PS4_BUTTON_BOOST_REVERSE", str(DEFAULT_PS4_BUTTON_BOOST_REVERSE))
)
PS4_BUTTON_BOOST_FORWARD = int(
    os.getenv("TP2_PS4_BUTTON_BOOST_FORWARD", str(DEFAULT_PS4_BUTTON_BOOST_FORWARD))
)
PS4_DEVICE_NAME_HINTS = tuple(
    token.strip().lower()
    for token in os.getenv(
        "TP2_PS4_DEVICE_HINTS",
        "wireless controller,dualshock,sony interactive entertainment,ps4",
    ).split(",")
    if token.strip()
)
INFERENCE_ENABLED = os.getenv("TP2_ENABLE_INFERENCE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
INFERENCE_INTERVAL_SEC = float(os.getenv("TP2_INFERENCE_INTERVAL_SEC", "0.75"))
INFERENCE_MIN_CONFIDENCE = float(os.getenv("TP2_INFERENCE_MIN_CONFIDENCE", "0.0"))
INFERENCE_TEMP_DIR = Path(
    os.getenv("TP2_INFERENCE_TEMP_DIR", "/tmp/tp2-car1-inference")
).expanduser()

server_address = (BIND_IP, BIND_PORT)
sock = None


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def is_neutral(accelerator, steering):
    return abs(accelerator) < 0.01 and abs(steering - STEERING_CENTER) < 0.01


class ControlState:
    def __init__(self):
        self.lock = threading.Lock()
        self.keyboard_accelerator = 0.0
        self.keyboard_steering = STEERING_CENTER
        self.ps4_accelerator = 0.0
        self.ps4_steering = STEERING_CENTER
        self.active_source = "keyboard"

    def set_keyboard(self, *, accelerator=None, steering=None):
        with self.lock:
            if accelerator is not None:
                self.keyboard_accelerator = accelerator
            if steering is not None:
                self.keyboard_steering = steering
            self.active_source = "keyboard"

    def set_ps4(self, accelerator, steering):
        with self.lock:
            self.ps4_accelerator = accelerator
            self.ps4_steering = steering
            if self.active_source == "ps4" or not is_neutral(accelerator, steering):
                self.active_source = "ps4"

    def ps4_disconnected(self):
        with self.lock:
            self.ps4_accelerator = 0.0
            self.ps4_steering = STEERING_CENTER
            if self.active_source == "ps4":
                self.active_source = "keyboard"

    def get_current(self):
        with self.lock:
            if self.active_source == "ps4":
                return self.ps4_steering, self.ps4_accelerator, self.active_source
            return self.keyboard_steering, self.keyboard_accelerator, self.active_source


class LiveInferenceWorker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.config = InferenceConfig.from_env()
        self.client = None
        self.status = "disabled" if not enabled else "starting"
        self.last_error = ""
        self.last_latency_sec = None
        self.last_completed_at = 0.0
        self.last_predictions: list[dict] = []
        self.last_result = None
        self.latest_frame = None
        self.latest_frame_id = 0
        self.processed_frame_id = 0
        self.next_run_at = 0.0

        if not enabled:
            return

        worker = threading.Thread(target=self._run, daemon=True)
        worker.start()

    def submit(self, frame):
        if not self.enabled:
            return

        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_frame_id += 1

    def render(self, frame):
        with self.lock:
            status = self.status
            last_error = self.last_error
            last_latency_sec = self.last_latency_sec
            last_completed_at = self.last_completed_at
            predictions = list(self.last_predictions)
            config = self.config

        annotated = draw_predictions_on_image(
            frame,
            predictions,
            min_confidence=INFERENCE_MIN_CONFIDENCE,
        )

        info_line = f"AI: {status} | mode={config.mode} | target={config.target} | det={len(predictions)}"
        age_line = "AI last result: waiting"
        if last_completed_at > 0:
            age_line = f"AI last result: {time.time() - last_completed_at:.1f}s ago"
        if last_latency_sec is not None:
            age_line += f" | latency={last_latency_sec:.2f}s"

        lines = [info_line, age_line]
        if last_error:
            lines.append(f"AI error: {last_error[:100]}")

        return annotated, lines

    def _run(self):
        while True:
            now = time.time()
            with self.lock:
                needs_frame = self.latest_frame_id == self.processed_frame_id
                must_wait = now < self.next_run_at
                if needs_frame or must_wait:
                    frame = None
                    frame_id = 0
                else:
                    frame = self.latest_frame.copy()
                    frame_id = self.latest_frame_id
                    self.status = "running"
                    self.next_run_at = now + INFERENCE_INTERVAL_SEC

            if frame is None:
                time.sleep(0.03)
                continue

            started_at = time.time()
            temp_image_path = None
            try:
                self.config.validate()
                if self.config.mode == "local" and not local_endpoint_reachable(self.config.api_url):
                    raise ConnectionError(
                        f"No hay endpoint local accesible en {self.config.api_url}"
                    )

                if self.client is None:
                    self.client = create_client(self.config)

                INFERENCE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    dir=INFERENCE_TEMP_DIR,
                    delete=False,
                ) as handle:
                    temp_image_path = Path(handle.name)

                ok = cv2.imwrite(str(temp_image_path), frame)
                if not ok:
                    raise RuntimeError(f"No se pudo escribir imagen temporal: {temp_image_path}")

                result = infer_one_image(self.client, temp_image_path, self.config)
                predictions = extract_predictions(result)
                latency_sec = time.time() - started_at

                with self.lock:
                    self.processed_frame_id = frame_id
                    self.last_predictions = predictions
                    self.last_result = result
                    self.last_latency_sec = latency_sec
                    self.last_completed_at = time.time()
                    self.last_error = ""
                    self.status = "ok"
            except Exception as exc:
                with self.lock:
                    self.processed_frame_id = frame_id
                    self.last_error = str(exc)
                    self.last_latency_sec = None
                    self.last_completed_at = time.time()
                    self.status = "error"
            finally:
                if temp_image_path is not None:
                    temp_image_path.unlink(missing_ok=True)


controls = ControlState()
inference_worker = LiveInferenceWorker(enabled=INFERENCE_ENABLED)


def on_press(key):
    if key == pynput_keyboard.KeyCode.from_char("2"):
        controls.set_keyboard(accelerator=1.0)
    if key == pynput_keyboard.KeyCode.from_char("w"):
        controls.set_keyboard(accelerator=0.6)
    if key == pynput_keyboard.KeyCode.from_char("s"):
        controls.set_keyboard(accelerator=-0.5)
    if key == pynput_keyboard.KeyCode.from_char("x"):
        controls.set_keyboard(accelerator=-0.9)
    if key == pynput_keyboard.KeyCode.from_char("a"):
        controls.set_keyboard(steering=1.0)
    if key == pynput_keyboard.KeyCode.from_char("d"):
        controls.set_keyboard(steering=-1.0)


def on_release(key):
    if key == pynput_keyboard.KeyCode.from_char("2"):
        controls.set_keyboard(accelerator=0.0)
    if key == pynput_keyboard.KeyCode.from_char("w"):
        controls.set_keyboard(accelerator=0.0)
    if key == pynput_keyboard.KeyCode.from_char("s"):
        controls.set_keyboard(accelerator=0.0)
    if key == pynput_keyboard.KeyCode.from_char("x"):
        controls.set_keyboard(accelerator=0.0)
    if key == pynput_keyboard.KeyCode.from_char("a"):
        controls.set_keyboard(steering=STEERING_CENTER)
    if key == pynput_keyboard.KeyCode.from_char("d"):
        controls.set_keyboard(steering=STEERING_CENTER)


def start_keyboard_listener():
    if pynput_keyboard is None:
        print(
            "Keyboard control disabled: no se pudo inicializar pynput. "
            f"Detalle: {PYNPUT_IMPORT_ERROR}"
        )
        return None

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def send_control(control_giro, control_acelerador, address):
    sock.sendto(
        struct.pack("c", bytes("C", "ascii"))
        + struct.pack("d", round(control_giro, 3))
        + struct.pack("d", round(control_acelerador, 3)),
        address,
    )


def normalize_abs_value(device, code, value):
    absinfo = device.absinfo(code)
    if absinfo is None or absinfo.max == absinfo.min:
        return 0.0
    midpoint = (absinfo.max + absinfo.min) / 2.0
    half_range = (absinfo.max - absinfo.min) / 2.0
    if half_range == 0:
        return 0.0
    return clamp((value - midpoint) / half_range, -1.0, 1.0)


def normalize_trigger_value(device, code, value):
    absinfo = device.absinfo(code)
    if absinfo is None or absinfo.max == absinfo.min:
        return 0.0
    return clamp((value - absinfo.min) / (absinfo.max - absinfo.min), 0.0, 1.0)


def steering_from_axis(axis_value):
    if abs(axis_value) < PS4_DEADZONE:
        return STEERING_CENTER
    if axis_value < 0:
        return STEERING_CENTER + (-axis_value * (1.0 - STEERING_CENTER))
    return STEERING_CENTER - (axis_value * (STEERING_CENTER + 1.0))


def accelerator_from_triggers(
    forward_value, reverse_value, estop_pressed, boost_forward_pressed, boost_reverse_pressed
):
    if estop_pressed:
        return 0.0
    if forward_value >= reverse_value:
        max_forward = 1.0 if boost_forward_pressed else 0.6
        return clamp(forward_value * max_forward, 0.0, max_forward)
    max_reverse = 0.9 if boost_reverse_pressed else 0.5
    return clamp(-(reverse_value * max_reverse), -max_reverse, 0.0)


def find_ps4_device():
    if not list_devices or not InputDevice:
        return None

    for path in list_devices():
        try:
            device = InputDevice(path)
        except OSError:
            continue

        device_name = (device.name or "").lower()
        if any(token in device_name for token in PS4_DEVICE_NAME_HINTS):
            return device
    return None


def watch_ps4_controller():
    if not InputDevice:
        print("PS4 support disabled: python-evdev is not installed.")
        return

    while True:
        device = find_ps4_device()
        if device is None:
            time.sleep(PS4_SCAN_INTERVAL_SEC)
            continue

        print(f"PS4 controller detected on {device.path}: {device.name}")
        axis_state = {
            PS4_AXIS_STEERING: 0.0,
            PS4_AXIS_L2: 0.0,
            PS4_AXIS_R2: 0.0,
        }
        estop_pressed = False
        boost_forward_pressed = False
        boost_reverse_pressed = False

        try:
            for event in device.read_loop():
                if event.type == ecodes.EV_ABS:
                    if event.code in axis_state:
                        if event.code == PS4_AXIS_STEERING:
                            axis_state[event.code] = normalize_abs_value(
                                device, event.code, event.value
                            )
                        else:
                            axis_state[event.code] = normalize_trigger_value(
                                device, event.code, event.value
                            )

                        controls.set_ps4(
                            accelerator_from_triggers(
                                axis_state[PS4_AXIS_R2],
                                axis_state[PS4_AXIS_L2],
                                estop_pressed,
                                boost_forward_pressed,
                                boost_reverse_pressed,
                            ),
                            steering_from_axis(axis_state[PS4_AXIS_STEERING]),
                        )
                elif event.type == ecodes.EV_KEY:
                    if event.code == PS4_BUTTON_ESTOP:
                        estop_pressed = bool(event.value)
                    if event.code == PS4_BUTTON_BOOST_FORWARD:
                        boost_forward_pressed = bool(event.value)
                    if event.code == PS4_BUTTON_BOOST_REVERSE:
                        boost_reverse_pressed = bool(event.value)
                    controls.set_ps4(
                        accelerator_from_triggers(
                            axis_state[PS4_AXIS_R2],
                            axis_state[PS4_AXIS_L2],
                            estop_pressed,
                            boost_forward_pressed,
                            boost_reverse_pressed,
                        ),
                        steering_from_axis(axis_state[PS4_AXIS_STEERING]),
                    )
        except OSError:
            print("PS4 controller disconnected.")
            controls.ps4_disconnected()

        time.sleep(PS4_SCAN_INTERVAL_SEC)


def draw_lidar_map(ranges):
    img_lidar = np.zeros((480, 640, 3), np.uint8)
    angle_index = 0
    for range_value in ranges:
        z_value = cmath.rect(range_value, math.radians(-angle_index - 90))
        angle_index = angle_index + 1
        if z_value.real != float("inf") and z_value.real != float("-inf"):
            x = int(z_value.real * 70) + 320
        else:
            x = 0
        if z_value.imag != float("inf") and z_value.imag != float("-inf"):
            y = int(z_value.imag * 70) + 240
        else:
            y = 0
        cv2.circle(img_lidar, (x, y), radius=2, color=(0, 0, 255), thickness=2)

    cv2.imshow("Mapa LIDAR", img_lidar)
    cv2.waitKey(1)


def main():
    global sock

    start_keyboard_listener()
    threading.Thread(target=watch_ps4_controller, daemon=True).start()

    received_payload = b""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)
    print(f"Manual control server listening on {server_address[0]}:{server_address[1]}")
    print("Keyboard: w/s/x/2 for throttle, a/d for steering")
    print("PS4: left stick to steer, R2 forward, L2 reverse, R1/L1 boost, X emergency neutral")
    print(
        "Inference: "
        f"{'enabled' if INFERENCE_ENABLED else 'disabled'} "
        f"({inference_worker.config.mode}/{inference_worker.config.target}) "
        f"endpoint={inference_worker.config.api_url}"
    )

    while True:
        data, address = sock.recvfrom(99999)
        data_type = struct.unpack("c", bytes([data[0]]))[0]
        received_payload = bytes(data[1:])
        data = pickle.loads(received_payload, encoding="latin1")

        if data_type == b"I":
            img = cv2.imdecode(data, 1)
            if img is None:
                continue

            inference_worker.submit(img)
            current_steering, current_accelerator, active_source = controls.get_current()
            annotated, inference_lines = inference_worker.render(img)

            overlay_lines = [
                f"Input: {active_source} | steer={current_steering:.2f} | throttle={current_accelerator:.2f}",
                *inference_lines,
            ]
            for idx, line in enumerate(overlay_lines):
                cv2.putText(
                    annotated,
                    line,
                    (10, 25 + (idx * 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if idx < 2 else (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Coche ARTEMIS", annotated)
            cv2.waitKey(1)
            send_control(current_steering, current_accelerator, address)
        if data_type == b"D":
            pass
        if data_type == b"L":
            draw_lidar_map(data)


if __name__ == "__main__":
    main()
