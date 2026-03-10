from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
import inference
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES


def build_app():
    # HttpInterface expects static assets at ./inference/landing/out relative to CWD.
    package_root = Path(inference.__file__).resolve().parent.parent
    os.chdir(package_root)

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if api_key:
        os.environ["ROBOFLOW_API_KEY"] = api_key

    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = ModelManager(registry)
    interface = HttpInterface(model_manager=manager)
    return interface.app


def main():
    parser = argparse.ArgumentParser(description="Start local Roboflow inference server without Docker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9001, type=int)
    args = parser.parse_args()

    app = build_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
