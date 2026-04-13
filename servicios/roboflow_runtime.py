from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import socket
from typing import Any
from urllib.parse import urlparse

import cv2

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None


def load_machine_env_file() -> Path | None:
    env_file_candidates = [
        os.getenv("TP2_INFERENCE_ENV_FILE", "").strip(),
        os.getenv("TP2_COCHE_ENV_FILE", "").strip(),
        "~/.config/tp2/inference.env",
        "~/.config/tp2/coche-jetson.env",
        "/etc/tp2/inference.env",
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

DEFAULT_MODE = os.getenv("TP2_INFERENCE_MODE", "local").strip().lower()
DEFAULT_TARGET = os.getenv("TP2_INFERENCE_TARGET", "").strip().lower()
DEFAULT_LOCAL_API_URL = os.getenv("ROBOFLOW_LOCAL_API_URL", "http://127.0.0.1:9001").strip()
DEFAULT_CLOUD_WORKFLOW_API_URL = os.getenv(
    "ROBOFLOW_CLOUD_WORKFLOW_API_URL", "https://serverless.roboflow.com"
).strip()
DEFAULT_CLOUD_MODEL_API_URL = os.getenv(
    "ROBOFLOW_CLOUD_MODEL_API_URL", "https://detect.roboflow.com"
).strip()
DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
DEFAULT_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "1-v8mk1").strip()
DEFAULT_WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "custom-workflow-2").strip()
DEFAULT_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "").strip()


def resolve_target(target: str, model_id: str) -> str:
    normalized_target = (target or "").strip().lower()
    if normalized_target in {"workflow", "model"}:
        return normalized_target
    return "model" if (model_id or "").strip() else "workflow"


@dataclass(frozen=True)
class InferenceConfig:
    mode: str
    target: str
    local_api_url: str
    cloud_workflow_api_url: str
    cloud_model_api_url: str
    api_key: str
    workspace: str
    workflow: str
    model_id: str

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        model_id = DEFAULT_MODEL_ID
        return cls(
            mode=DEFAULT_MODE if DEFAULT_MODE in {"local", "cloud"} else "local",
            target=resolve_target(DEFAULT_TARGET, model_id),
            local_api_url=DEFAULT_LOCAL_API_URL,
            cloud_workflow_api_url=DEFAULT_CLOUD_WORKFLOW_API_URL,
            cloud_model_api_url=DEFAULT_CLOUD_MODEL_API_URL,
            api_key=DEFAULT_API_KEY,
            workspace=DEFAULT_WORKSPACE,
            workflow=DEFAULT_WORKFLOW,
            model_id=model_id,
        )

    @property
    def api_url(self) -> str:
        return select_api_url(
            mode=self.mode,
            target=self.target,
            local_api_url=self.local_api_url,
            cloud_workflow_api_url=self.cloud_workflow_api_url,
            cloud_model_api_url=self.cloud_model_api_url,
        )

    def validate(self):
        if self.mode not in {"local", "cloud"}:
            raise ValueError("TP2_INFERENCE_MODE debe ser local o cloud.")
        if self.target not in {"workflow", "model"}:
            raise ValueError("TP2_INFERENCE_TARGET debe ser workflow o model.")
        if self.target == "workflow":
            if not self.workspace:
                raise ValueError("Falta ROBOFLOW_WORKSPACE para workflow.")
            if not self.workflow:
                raise ValueError("Falta ROBOFLOW_WORKFLOW para workflow.")
        if self.target == "model" and not self.model_id:
            raise ValueError(
                "Para TP2_INFERENCE_TARGET=model debes definir ROBOFLOW_MODEL_ID, "
                "por ejemplo tu-proyecto/1."
            )


def local_endpoint_reachable(api_url: str, timeout_sec: float = 2.0) -> bool:
    parsed = urlparse(api_url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def create_client(config: InferenceConfig):
    if InferenceHTTPClient is None:
        raise ImportError(
            "No se pudo importar inference_sdk. Instala la dependencia en el entorno activo."
        )
    return InferenceHTTPClient(api_url=config.api_url, api_key=config.api_key)


def select_api_url(
    mode: str,
    target: str,
    local_api_url: str,
    cloud_workflow_api_url: str,
    cloud_model_api_url: str,
) -> str:
    if mode == "local":
        return local_api_url.strip()
    if target == "workflow":
        return cloud_workflow_api_url.strip()
    return cloud_model_api_url.strip()


def infer_one_image(client, image_path: Path, config: InferenceConfig):
    if config.target == "workflow":
        return client.run_workflow(
            workspace_name=config.workspace,
            workflow_id=config.workflow,
            images={"image": str(image_path)},
            use_cache=True,
        )

    if config.target == "model":
        return client.infer(str(image_path), model_id=config.model_id)

    raise ValueError("TP2_INFERENCE_TARGET debe ser workflow o model.")


def extract_predictions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        merged: list[dict[str, Any]] = []
        for item in payload:
            merged.extend(extract_predictions(item))
        return merged

    if not isinstance(payload, dict):
        return []

    top_predictions = payload.get("predictions")
    if isinstance(top_predictions, list):
        return top_predictions

    if isinstance(top_predictions, dict):
        nested_predictions = top_predictions.get("predictions")
        if isinstance(nested_predictions, list):
            return nested_predictions

    return []


def draw_predictions_on_image(
    image,
    predictions: list[dict[str, Any]],
    *,
    min_confidence: float = 0.0,
):
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for prediction in predictions:
        confidence = prediction.get("confidence")
        if isinstance(confidence, (float, int)) and confidence < min_confidence:
            continue

        x = prediction.get("x")
        y = prediction.get("y")
        w = prediction.get("width")
        h = prediction.get("height")
        if None in (x, y, w, h):
            continue

        x1 = max(0, int(round(x - w / 2)))
        y1 = max(0, int(round(y - h / 2)))
        x2 = min(img_w - 1, int(round(x + w / 2)))
        y2 = min(img_h - 1, int(round(y + h / 2)))

        label = str(prediction.get("class", "unknown"))
        text = f"{label} {confidence:.2f}" if isinstance(confidence, (float, int)) else label

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y1 = max(0, y1 - text_h - baseline - 6)
        text_y2 = max(text_h + baseline + 6, y1)
        text_x2 = min(img_w - 1, x1 + text_w + 8)

        cv2.rectangle(annotated, (x1, text_y1), (text_x2, text_y2), (0, 220, 0), -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 4, text_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def draw_predictions(
    image_path: Path,
    output_path: Path,
    predictions: list[dict[str, Any]],
    *,
    min_confidence: float = 0.0,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    annotated = draw_predictions_on_image(
        image,
        predictions,
        min_confidence=min_confidence,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), annotated)
    if not ok:
        raise RuntimeError(f"No se pudo escribir la imagen de salida: {output_path}")
